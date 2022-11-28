"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from ImageNetV2:
# https://github.com/modestyachts/ImageNetV2
# --------------------------------------------------------

import os


from torchvision.datasets import ImageFolder


class ImageNetV2(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(
            root=os.path.join(
                root, "imagenetv2/imagenetv2-matched-frequency-format-val"
            ),
            transform=transform,
        )

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
        classes = [str(i) for i in range(1000)]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
