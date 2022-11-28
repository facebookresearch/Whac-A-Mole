"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from lvis import LVIS
from skimage import measure


BBOX_AREA_THRESHOLD = 3600


class LVISDataset(Dataset):
    lvis_base_folder = "lvis"
    coco_base_folder = "coco"

    def __init__(
        self,
        root,
        cat_id,
        transform=None,
        threshold_standard="bbox",
        area_thresh=BBOX_AREA_THRESHOLD,
    ):
        super(LVISDataset, self).__init__()
        self.transform = transform

        # filter based on bbox area to keep large objects
        filtered_ann_list = []
        split_to_meta_data = {}

        for split in ["train", "val"]:
            lvis_meta_data = LVIS(
                os.path.join(
                    root, self.lvis_base_folder, f"lvis_v1_{split}.json"
                )
            )
            split_to_meta_data[split] = lvis_meta_data

            anns = lvis_meta_data.dataset["annotations"]
            for ann in anns:
                if ann["category_id"] != cat_id:
                    continue

                if threshold_standard == "bbox":
                    w, h = ann["bbox"][2:]
                    bbox_area = w * h
                    if bbox_area < area_thresh:
                        continue
                else:
                    assert threshold_standard == "mask"
                    if ann["area"] < area_thresh:
                        continue

                mask = lvis_meta_data.ann_to_mask(ann)
                num_components = measure.label(
                    mask, background=0, return_num=True
                )[1]
                if num_components > 1:
                    continue

                filtered_ann_list.append(ann)

        self.split_to_meta_data = split_to_meta_data

        train_split_img_root = os.path.join(
            root, self.coco_base_folder, "train2017"
        )
        val_split_img_root = os.path.join(
            root, self.coco_base_folder, "val2017"
        )
        assert os.path.exists(train_split_img_root)
        assert os.path.exists(val_split_img_root)

        self.split_to_img_root = {
            "train": train_split_img_root,
            "val": val_split_img_root,
        }
        self.ann_list = filtered_ann_list

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, index):
        ann = self.ann_list[index]
        img_id = ann["image_id"]
        img_fname = f"{img_id:012d}.jpg"

        train_img_path = os.path.join(
            self.split_to_img_root["train"], img_fname
        )
        val_img_path = os.path.join(self.split_to_img_root["val"], img_fname)
        if os.path.exists(train_img_path):
            img_path = train_img_path
        elif os.path.exists(val_img_path):
            img_path = val_img_path
        else:
            raise NotImplementedError
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        img_id = ann["image_id"]
        if img_id in self.split_to_meta_data["train"].imgs:
            mask = self.split_to_meta_data["train"].ann_to_mask(ann)
        elif img_id in self.split_to_meta_data["val"].imgs:
            mask = self.split_to_meta_data["val"].ann_to_mask(ann)
        else:
            raise NotImplementedError

        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        return img, mask
