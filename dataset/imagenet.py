"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from TorchVision:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py
# --------------------------------------------------------

import os
import torch


from typing import Any
from torchvision.datasets.folder import ImageFolder


class ImageNet(ImageFolder):
    base_folder = "imagenet"

    def __init__(
        self,
        root: str,
        split: str = "train",
        return_group_index=False,
        return_file_path=False,
        return_dist_shift_index=False,
        return_image_size=False,
        dist_shift_index=0,
        **kwargs: Any
    ) -> None:
        assert split in ["train", "val"]
        root = self.root = os.path.join(root, self.base_folder)
        self.split = split
        wnid_to_classes = self.load_meta_file(self.root)

        super().__init__(self.split_folder, **kwargs)
        self.root = root
        self.return_group_index = return_group_index
        self.return_file_path = return_file_path
        self.return_dist_shift_index = return_dist_shift_index
        self.return_image_size = return_image_size
        self.dist_shift_index = dist_shift_index

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self, root):
        fpath = os.path.join(root, "labels.txt")
        with open(fpath, "r") as f:
            lines = f.readlines()

        wnid_to_classes = {}

        for line in lines:
            wn_id, cls_name = line.strip().split(",")
            wnid_to_classes[wn_id] = cls_name

        return wnid_to_classes

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        if self.return_image_size:
            img_fpath = self.samples[index][0]
            img = self.loader(img_fpath)
            data_dict["img_size"] = torch.tensor(img.size)

        if self.return_group_index:
            data_dict["group_index"] = self.group_array[index]

        if self.return_file_path:
            data_dict["file_path"] = self.imgs[index][0]

        if self.return_dist_shift_index:
            data_dict["dist_shift"] = self.dist_shift_index

        return data_dict

    def set_num_group_and_group_array(self, num_shortcut_cat, shortcut_label):
        self.num_group = len(self.classes) * num_shortcut_cat
        self.group_array = (
            torch.tensor(self.targets, dtype=torch.long) * num_shortcut_cat
            + shortcut_label
        )

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights


def get_imagenet_class_name_list():
    with open("data/imagenet/labels.txt") as f:
        lines = f.readlines()

    prefix_len = len("n02892201,")
    class_name_list = [line[prefix_len:].strip() for line in lines]
    return class_name_list
