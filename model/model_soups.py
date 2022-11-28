"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Model Soups:
# https://github.com/mlfoundations/model-soups
# --------------------------------------------------------

import torch
import math


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    return model
