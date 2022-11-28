"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Model Soups:
# https://github.com/mlfoundations/model-soups/blob/main/datasets/objectnet.py
# --------------------------------------------------------

import os
import json
import torch
import numpy as np


from torchvision.datasets import ImageFolder


def objectnet_crop(img):
    width, height = img.size
    cropArea = (2, 2, width - 2, height - 2)
    img = img.crop(cropArea)
    return img


class ObjectNet(ImageFolder):
    base_folder = "objectnet-1.0"

    def __init__(self, root, transform):
        root = os.path.join(root, self.base_folder)
        self.root = root
        (
            self._class_sublist,
            self.class_sublist_mask,
            self.folders_to_ids,
            self.classname_map,
        ) = self.get_metadata(root)

        self.label_map = {
            name: idx
            for idx, name in enumerate(sorted(list(self.folders_to_ids.keys())))
        }
        image_dir = os.path.join(root, "images")
        super().__init__(image_dir, transform=transform)

        self.samples = [
            d
            for d in self.samples
            if os.path.basename(os.path.dirname(d[0])) in self.label_map
        ]
        self.imgs = self.samples

        self.classnames = sorted(list(self.folders_to_ids.keys()))
        self.rev_class_idx_map = {}
        self.class_idx_map = {}
        for idx, name in enumerate(self.classnames):
            self.rev_class_idx_map[idx] = self.folders_to_ids[name]
            for imagenet_idx in self.rev_class_idx_map[idx]:
                self.class_idx_map[imagenet_idx] = idx

        self.crop = objectnet_crop
        self.classnames = [
            self.classname_map[c].lower() for c in self.classnames
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        sample = self.crop(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        label = os.path.basename(os.path.dirname(path))
        return {
            'image': sample,
            'target': self.label_map[label],
        }

    def map_prediction(self, logits):
        device = logits.device
        if logits.shape[1] == 113:
            return logits
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        logits_projected = np.zeros((logits.shape[0], 113))
        for k, v in self.rev_class_idx_map.items():
            logits_projected[:, k] = np.max(logits[:, v], axis=1).squeeze()
        return torch.tensor(logits_projected).to(device)

    def get_metadata(self, root):
        metadata = os.path.join(root, "mappings")

        with open(
            os.path.join(metadata, "folder_to_objectnet_label.json"), "r"
        ) as f:
            folder_map = json.load(f)
            folder_map = {v: k for k, v in folder_map.items()}

        with open(
            os.path.join(metadata, "objectnet_to_imagenet_1k.json"), "r"
        ) as f:
            objectnet_map = json.load(f)

        with open(
            os.path.join(metadata, "pytorch_to_imagenet_2012_id.json"), "r"
        ) as f:
            pytorch_map = json.load(f)
            pytorch_map = {v: k for k, v in pytorch_map.items()}

        with open(
            os.path.join(metadata, "imagenet_to_label_2012_v2"), "r"
        ) as f:
            imagenet_map = {
                v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)
            }

        folder_to_ids, class_sublist = {}, []
        for objectnet_name, imagenet_names in objectnet_map.items():
            imagenet_names = imagenet_names.split("; ")
            imagenet_ids = [
                int(imagenet_map[imagenet_name])
                for imagenet_name in imagenet_names
            ]
            class_sublist.extend(imagenet_ids)
            folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

        class_sublist = sorted(class_sublist)
        class_sublist_mask = [(i in class_sublist) for i in range(1000)]
        classname_map = {v: k for k, v in folder_map.items()}

        return class_sublist, class_sublist_mask, folder_to_ids, classname_map
