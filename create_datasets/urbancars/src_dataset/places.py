"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os


from torch.utils.data import Dataset
from PIL import Image


class Places(Dataset):
    base_folder = "places"
    filter_objs_metadata_folder = "create_datasets/urbancars/metadata/places/filter_objs"

    def __init__(self, root, category_name, transform=None):
        super().__init__()
        self.root = root

        rel_fpath_list = []
        for split in ["train", "val"]:
            split_dir = os.path.join(root, self.base_folder, split, category_name)
            assert os.path.exists(split_dir)
            metadata_fpath = os.path.join(
                self.filter_objs_metadata_folder, split, f"{category_name}.txt"
            )
            assert os.path.exists(metadata_fpath), f"{metadata_fpath} does not exist"
            with open(metadata_fpath, "r") as f:
                filename_list = [fname.strip() for fname in f.readlines()]
            filename_list.sort()
            split_rel_fpath_list = [
                os.path.join(split, category_name, filename)
                for filename in filename_list
            ]
            rel_fpath_list += split_rel_fpath_list
        self.rel_fpath_list = rel_fpath_list
        self.transform = transform

    def __len__(self):
        return len(self.rel_fpath_list)

    def __getitem__(self, index):
        rel_fpath = self.rel_fpath_list[index]
        img_path = os.path.join(self.root, self.base_folder, rel_fpath)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
