"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from ImageNet-D:
# https://github.com/bethgelab/robustness/blob/main/examples/imagenet_d/map_files.py
# --------------------------------------------------------

import os


from torchvision.datasets import ImageFolder
from dataset.imagenet_d_utils import get_imagenet_visda_mapping_quick


class ImageNetD(ImageFolder):
    def __init__(self, root, dataset, transform=None):
        assert dataset.startswith("imagenet-d")
        dataset = dataset[len("imagenet-d-"):]
        assert dataset in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        super().__init__(root=os.path.join(root, f"imagenet-d/{dataset}"), transform=transform)

        self.mapping = get_imagenet_visda_mapping_quick()[0]

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        return data_dict

    def map_prediction(self, pred):
        return self.mapping[pred.to(self.mapping.device)]
