"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import open_clip
import torch.nn.functional as F


from .clip_imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from tqdm import tqdm


def get_classifier(arch, num_classes, weights="IMAGENET1K_V1", new_fc=True):
    if arch.startswith("resnet"):
        model = models.__dict__[arch](weights=weights)
        if new_fc:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError

    return model


def get_transforms(arch, is_training):
    if arch.startswith("resnet"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    else:
        raise NotImplementedError

    return transform


class DomainIndependentClassifier(nn.Module):
    def __init__(self, arch, num_classes, num_domain, weights="IMAGENET1K_V1"):
        super(DomainIndependentClassifier, self).__init__()
        self.backbone = get_classifier(arch, num_classes, weights=weights)
        self.domain_classifier_list = nn.ModuleList(
            [
                nn.Linear(self.backbone.fc.in_features, num_classes)
                for _ in range(num_domain)
            ]
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        logits_per_domain = [
            classifier(x) for classifier in self.domain_classifier_list
        ]
        logits_per_domain = torch.stack(logits_per_domain, dim=1)

        if self.training:
            return logits_per_domain
        else:
            return logits_per_domain.mean(dim=1)


class LastLayerEnsemble(nn.Module):
    def __init__(
        self,
        num_classes,
        num_dist_shift,
        backbone=None,
        in_features=None,
    ) -> None:
        super().__init__()
        assert not (backbone is not None and in_features is not None)
        if backbone is not None:
            self.backbone = backbone
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif in_features is not None:
            self.backbone = None
        else:
            raise ValueError("Either backbone or in_features must be provided")

        self.ensemble_classifier_list = nn.ModuleList(
            [nn.Linear(in_features, num_classes) for _ in range(num_dist_shift)]
        )

        self.dist_shift_predictor = nn.Linear(in_features, num_dist_shift)

    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)

        logits_per_dist_shift = [
            classifier(x) for classifier in self.ensemble_classifier_list
        ]
        logits_per_dist_shift = torch.stack(logits_per_dist_shift, dim=1)

        if self.training:
            dist_shift = self.dist_shift_predictor(x)
            return logits_per_dist_shift, dist_shift
        else:
            dist_shift = F.softmax(self.dist_shift_predictor(x), dim=1)
            return (logits_per_dist_shift * dist_shift.unsqueeze(-1)).sum(dim=1)


class CLIPIN1KZeroShotClassifier(nn.Module):
    def __init__(self, clip_model, device, use_amp=False):
        super().__init__()
        self.clip_model = clip_model

        # from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
        zeroshot_weights = []

        for classname in tqdm(imagenet_classnames, desc="build CLIP IN1K classifier"):
            texts = [
                template(classname) for template in openai_imagenet_template
            ]  # format with class
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    class_embeddings = clip_model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(
                        dim=0
                    )
                    class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ self.zeroshot_weights
        return logits


class CLIPIN1KFTClassifier(torch.nn.Module):
    def __init__(self, clip_model, feature_dim, num_classes, device, use_amp=False):
        super(CLIPIN1KFTClassifier, self).__init__()
        self.clip_model = clip_model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)

        zero_shot_weights = self._build_zero_shot_weights(device, use_amp)
        with torch.no_grad():
            self.classification_head.weight = torch.nn.Parameter(zero_shot_weights.clone())
            self.classification_head.bias.zero_()

        # Note: modified. Get rid of the language part.
        if hasattr(self.clip_model, 'transformer'):
            delattr(self.clip_model, 'transformer')

    def _build_zero_shot_weights(self, device, use_amp):
        zeroshot_weights = []

        for classname in tqdm(imagenet_classnames, desc="build CLIP IN1K classifier"):
            texts = [
                template(classname) for template in openai_imagenet_template
            ]  # format with class
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    class_embeddings = self.clip_model.encode_text(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(
                        dim=0
                    )
                    class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return 100 * zeroshot_weights.t()

    def forward(self, images, return_features=False):
        features = self.clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits
