"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch


from torch.utils.data import Dataset, default_collate
from PIL import Image


class ImageNetSal(Dataset):
    base_folder = "imagenet_sal"
    imagenet_base_folder = "imagenet"

    def __init__(
        self,
        root,
        split,
        co_transform,
        bg_transform,
        final_transform,
        mask_final_transform,
        return_dist_shift=False,
        dist_shift_index=2,
    ):
        self.root = os.path.join(root, self.base_folder, split)
        meta_data_file_path = os.path.join(
            root, self.base_folder, f"{split}_meta_data.txt"
        )
        self.co_transform = co_transform
        self.final_transform = final_transform
        self.mask_final_transform = mask_final_transform
        self.bg_transform = bg_transform
        self.classes, self.class_to_idx = self.find_classes()
        (
            self.file_path_list,
            self.target_list,
            self.file_path_list_per_class,
        ) = self.find_image_and_target(meta_data_file_path)

        self.return_dist_shift = return_dist_shift
        self.dist_shift_index = dist_shift_index

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, index):
        file_path_no_ext = os.path.join(self.root, self.file_path_list[index])
        target = self.target_list[index]

        mask_file_path = f"{file_path_no_ext}_fg_mask_.png"
        fg_file_path = f"{file_path_no_ext}_fg_.png"
        bg_file_path = self.find_random_background(target)

        fg_image = Image.open(fg_file_path).convert("RGB")
        mask_image = Image.open(mask_file_path).convert("L")
        bg_image = Image.open(bg_file_path).convert("RGB")

        fg_image, mask_image = self.co_transform((fg_image, mask_image))
        fg_image = self.final_transform(fg_image)
        mask_image = self.mask_final_transform(mask_image)
        bg_image = self.bg_transform(bg_image)

        data_dict = {
            "fg_image": fg_image,
            "bg_image": bg_image,
            "mask": mask_image,
            "target": target,
        }

        if self.return_dist_shift:
            data_dict["dist_shift"] = self.dist_shift_index

        return data_dict

    def find_classes(self):
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

    def find_image_and_target(self, meta_data_file_path):
        file_path_list, target_list = [], []
        file_path_list_per_class = [[] for _ in range(len(self.classes))]

        with open(meta_data_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            rel_file_path, target = line.split(",")
            target = int(target)
            file_path_list.append(rel_file_path)
            target_list.append(target)

            file_path_list_per_class[target].append(rel_file_path)

        return file_path_list, target_list, file_path_list_per_class

    def find_random_background(self, target):
        while True:
            bg_target = torch.randint(0, 1000, (1,)).item()
            if bg_target != target:
                break

        bg_cls_file_path_list = self.file_path_list_per_class[bg_target]
        rand_idx = torch.randint(0, len(bg_cls_file_path_list), (1,)).item()
        bg_file_path = os.path.join(
            self.root, f"{bg_cls_file_path_list[rand_idx]}_bg_.png"
        )
        return bg_file_path


def imagenet_sal_collate(batch):
    in_1k_batch_list = []
    in_mixed_rand_batch_list = []

    for data_dict in batch:
        if "image" in data_dict:
            in_1k_batch_list.append(data_dict)
        else:
            in_mixed_rand_batch_list.append(data_dict)

    final_batch_dict = {}

    if len(in_1k_batch_list) > 0:
        in_1k_batch = default_collate(in_1k_batch_list)
        final_batch_dict.update(in_1k_batch)

    if len(in_mixed_rand_batch_list) > 0:
        in_mixed_rand_batch = default_collate(in_mixed_rand_batch_list)
        for k, v in in_mixed_rand_batch.items():
            if k == "target":
                k = "bg_aug_target"
            if k == "dist_shift":
                k = "bg_aug_dist_shift"
            final_batch_dict[k] = v

    return final_batch_dict

