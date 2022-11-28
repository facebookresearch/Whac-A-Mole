"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from ImageNet-Sketch:
# https://github.com/HaohanWang/ImageNet-Sketch
# --------------------------------------------------------

import os


from torchvision.datasets import ImageFolder


class ImageNetSketch(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=os.path.join(root, "imagenet-sketch"), transform=transform)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        return data_dict

    def map_prediction(self, pred):
        return pred

    def find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """

        with open("data/imagenet/labels.txt", "r") as f:
            lines = f.readlines()

        classes = []
        for line in lines:
            line = line.strip()
            wn_id = line.split(",")[0]
            classes.append(wn_id)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
