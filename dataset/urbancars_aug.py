"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import torch


from PIL import Image
from torch.utils.data import Dataset, default_collate


class UrbanCarsAug(Dataset):
    base_folder = "urbancars"

    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    aug_mode_to_dist_shift = {
        "bg": 1,
        "co_occur_obj": 2,
        "both": 3,
    }

    def __init__(
        self,
        root: str,
        aug_mode: str,
        co_transform,
        bg_transform,
        final_transform,
        mask_final_transform,
        group_label="both",
    ):
        bg_ratio = co_occur_obj_ratio = 0.95
        assert os.path.exists(os.path.join(root, self.base_folder))

        super().__init__()
        assert aug_mode in ["bg", "co_occur_obj", "both"]
        assert group_label in ["bg", "co_occur_obj", "both"]
        self.co_transform = co_transform
        self.bg_transform = bg_transform
        self.final_transform = final_transform
        self.mask_final_transform = mask_final_transform

        self.bg_ratio = bg_ratio
        self.co_occur_obj_ratio = co_occur_obj_ratio
        self.aug_mode = aug_mode
        self.aug_dist_shift_label = self.aug_mode_to_dist_shift[aug_mode]

        ratio_combination_folder_name = (
            f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        )
        img_root = os.path.join(
            root, self.base_folder, ratio_combination_folder_name, "train"
        )

        self.img_fpath_list = []
        self.obj_bg_co_occur_obj_label_list = []

        self.count_per_intersectional_group = torch.zeros(
            len(self.obj_name_list),
            len(self.bg_name_list),
            len(self.obj_name_list),
            dtype=torch.long,
        )

        obj_id_to_img_fpath_list = []
        for _ in range(len(self.obj_name_list)):
            obj_id_to_img_fpath_list.append([])

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):

                for co_occur_obj_id, co_occur_obj_name in enumerate(
                    self.co_occur_obj_name_list
                ):
                    dir_name = f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    dir_path = os.path.join(img_root, dir_name)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(
                        os.path.join(dir_path, "*.jpg")
                    )
                    self.img_fpath_list += img_fpath_list
                    self.count_per_intersectional_group += len(
                        img_fpath_list
                    )
                    obj_id_to_img_fpath_list[obj_id] += img_fpath_list

                    self.obj_bg_co_occur_obj_label_list += [
                        (obj_id, bg_id, co_occur_obj_id)
                    ] * len(img_fpath_list)

        self.obj_id_to_img_fpath_list = obj_id_to_img_fpath_list

        self.obj_bg_co_occur_obj_label_list = torch.tensor(
            self.obj_bg_co_occur_obj_label_list, dtype=torch.long
        )

        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]

        if group_label == "bg":
            num_shortcut_cat = 2
            shortcut_label = bg_label
        elif group_label == "co_occur_obj":
            num_shortcut_cat = 2
            shortcut_label = co_occur_obj_label
        elif group_label == "both":
            num_shortcut_cat = 4
            shortcut_label = bg_label * 2 + co_occur_obj_label
        else:
            raise NotImplementedError

        self.domain_label = shortcut_label
        self.set_num_group_and_group_array(num_shortcut_cat, shortcut_label)

    def set_num_group_and_group_array(self, num_shortcut_cat, shortcut_label):
        self.num_group = len(self.obj_name_list) * num_shortcut_cat
        self.group_array = self.obj_label * num_shortcut_cat + shortcut_label

    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self.img_fpath_list[index]
        label = self.obj_bg_co_occur_obj_label_list[index]

        img = Image.open(img_fpath)
        img = img.convert("RGB")

        # mask
        obj_mask = self._get_obj_mask_by_img_fpath(img_fpath)
        img, obj_mask = self.co_transform((img, obj_mask))
        img = self.final_transform(img)
        obj_mask = self.mask_final_transform(obj_mask)

        obj_label = label[0]
        img_fpath_for_bg = self._sample_img_fpath_for_bg(obj_label)
        bg_img = self._get_bg_img_by_img_fpath(img_fpath_for_bg)
        bg_img = self.bg_transform(bg_img)

        img_fpath_for_co_occur_obj = self._sample_img_fpath_for_co_occur_obj(
            obj_label
        )
        (
            co_occur_obj_img,
            co_occur_obj_mask,
        ) = self._get_co_occur_obj_img_and_mask_by_img_fpath(
            img_fpath_for_co_occur_obj
        )
        co_occur_obj_img, co_occur_obj_mask = self.co_transform((co_occur_obj_img, co_occur_obj_mask))
        co_occur_obj_img = self.final_transform(co_occur_obj_img)
        co_occur_obj_mask = self.mask_final_transform(co_occur_obj_mask)

        data_dict = {
            "aug_image": img,
            "aug_bg_image": bg_img,
            "aug_co_occur_obj_image": co_occur_obj_img,
            "aug_obj_mask": obj_mask,
            "aug_co_occur_obj_mask": co_occur_obj_mask,
            "aug_label": label,
            "aug_dist_shift": self.aug_dist_shift_label,
        }

        return data_dict

    def _get_obj_mask_by_img_fpath(self, img_fpath):
        fpath_wo_ext = os.path.splitext(img_fpath)[0]
        obj_mask_fpath = fpath_wo_ext + "_mask.png"
        obj_mask = Image.open(obj_mask_fpath).convert("L")
        return obj_mask

    def _get_bg_img_by_img_fpath(self, img_fpath):
        fpath_wo_ext = os.path.splitext(img_fpath)[0]
        bg_only_fpath = fpath_wo_ext + "_bg_only.png"
        bg_only_img = Image.open(bg_only_fpath).convert("RGB")
        return bg_only_img

    def _get_co_occur_obj_img_and_mask_by_img_fpath(self, img_fpath):
        img_for_co_occur_obj = Image.open(img_fpath).convert("RGB")

        fpath_wo_ext = os.path.splitext(img_fpath)[0]
        co_occur_obj_mask_fpath = fpath_wo_ext + "_co_occur_obj_mask.png"
        co_occur_obj_mask = Image.open(co_occur_obj_mask_fpath).convert("L")

        return img_for_co_occur_obj, co_occur_obj_mask

    def _sample_img_fpath_for_bg(self, obj_label):
        if self.aug_mode == "co_occur_obj":
            obj_label_for_bg = obj_label
        elif self.aug_mode in ["bg", "both"]:
            obj_label_for_bg = (obj_label + 1) % len(self.obj_name_list)
        else:
            raise NotImplementedError

        img_fpath_list = self.obj_id_to_img_fpath_list[obj_label_for_bg]
        rand_idx = torch.randint(0, len(img_fpath_list), (1,)).item()
        img_fpath = img_fpath_list[rand_idx]
        return img_fpath

    def _sample_img_fpath_for_co_occur_obj(self, obj_label):
        if self.aug_mode == "bg":
            obj_label_for_co_occur_obj = obj_label
        elif self.aug_mode in ["co_occur_obj", "both"]:
            obj_label_for_co_occur_obj = (obj_label + 1) % len(
                self.obj_name_list
            )
        else:
            raise NotImplementedError

        img_fpath_list = self.obj_id_to_img_fpath_list[
            obj_label_for_co_occur_obj
        ]
        rand_idx = torch.randint(0, len(img_fpath_list), (1,)).item()
        img_fpath = img_fpath_list[rand_idx]
        return img_fpath


def urbancars_aug_collate(batch):
    original_batch_list = []
    aug_batch_list = []

    for data_dict in batch:
        if "aug_image" in data_dict:
            aug_batch_list.append(data_dict)
        else:
            original_batch_list.append(data_dict)

    final_batch_dict = {}

    if len(original_batch_list) > 0:
        no_aug_batch = default_collate(original_batch_list)
        final_batch_dict.update(no_aug_batch)

    if len(aug_batch_list) > 0:
        aug_batch = default_collate(aug_batch_list)
        for k, v in aug_batch.items():
            assert k not in final_batch_dict
            final_batch_dict[k] = v

    return final_batch_dict
