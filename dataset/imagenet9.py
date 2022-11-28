"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from ImageNet-9 (Background Challenge):
# https://github.com/MadryLab/backgrounds_challenge
# --------------------------------------------------------

import json
import os
import torch


from torchvision.datasets import ImageFolder


class ImageNet9(ImageFolder):
    def __init__(self, root, dataset, transform=None):
        assert dataset in [
            "mixed_rand",
            "mixed_same",
        ]
        super().__init__(
            root=os.path.join(root, f"imagenet_9/{dataset}/val"),
            transform=transform,
        )

        with open(os.path.join(root, "imagenet_9/in_to_in9.json"), "r") as f:
            in_to_in9 = json.load(f)

        new_in_to_in9 = torch.ones(1000, dtype=torch.long) * -1
        for in_idx, in9_idx in in_to_in9.items():
            new_in_to_in9[int(in_idx)] = in9_idx

        self.indices_in_1k = new_in_to_in9

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        return data_dict

    def map_prediction(self, pred):
        mapped_pred = self.indices_in_1k[pred.to(self.indices_in_1k.device)]
        return mapped_pred
