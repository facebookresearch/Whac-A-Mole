"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import glob
import numpy as np


from PIL import Image
from tqdm import tqdm


def gen_bg_only(data_root):
    assert os.path.exists(data_root)
    for cue_combination_folder in os.listdir(data_root):
        src_folder = os.path.join(data_root, cue_combination_folder)

        for img_fpath in tqdm(
            glob.glob(os.path.join(src_folder, "*.jpg")),
            desc=cue_combination_folder,
        ):
            fpath_wo_ext = os.path.splitext(img_fpath)[0]
            bg_only_fpath = fpath_wo_ext + "_bg_only.png"
            if os.path.exists(bg_only_fpath):
                continue

            obj_mask_fpath = fpath_wo_ext + "_mask.png"
            co_occur_obj_mask_fpath = fpath_wo_ext + "_co_occur_obj_mask.png"
            img = np.array(Image.open(img_fpath).convert("RGB"))
            h, w = img.shape[:2]

            obj_mask = Image.open(obj_mask_fpath).convert("L")
            co_occur_obj_mask = Image.open(co_occur_obj_mask_fpath).convert("L")

            obj_mask = np.array(obj_mask).astype(np.int64)
            co_occur_obj_mask = np.array(co_occur_obj_mask).astype(np.int64)

            bg_crop_for_obj = img[
                int(0.25 * h) : int(0.75 * h), : int(0.25 * w)
            ]
            bg_crop_for_co_occur_obj = img[
                int(0.375 * h) : int(0.625 * h), : int(0.25 * w)
            ]

            bg_only = np.copy(img)
            bg_only[
                int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.5 * w)
            ] = bg_crop_for_obj
            bg_only[
                int(0.25 * h) : int(0.75 * h), int(0.5 * w) : int(0.75 * w)
            ] = bg_crop_for_obj
            bg_only[
                int(0.375 * h) : int(0.625 * h), int(0.75 * w) :
            ] = bg_crop_for_co_occur_obj

            Image.fromarray(bg_only).save(bg_only_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/urbancars/bg-0.95_co_occur_obj-0.95/train")
    args = parser.parse_args()
    gen_bg_only(args.data_root)
