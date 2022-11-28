"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os


from torchvision.datasets import ImageFolder


class ImageNetStylized(ImageFolder):
    def __init__(self, root, split, transform=None, return_dist_shift_index=False, dist_shift_index=1):
        super().__init__(root=os.path.join(root, "imagenet-stylized", split), transform=transform)
        self.return_dist_shift_index = return_dist_shift_index
        self.dist_shift_index = dist_shift_index

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        if self.return_dist_shift_index:
            data_dict["dist_shift"] = self.dist_shift_index

        return data_dict

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

    def map_prediction(self, pred):
        return pred
