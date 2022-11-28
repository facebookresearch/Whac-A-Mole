"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os


from tqdm import tqdm


def build_imagenet_sal_meta_data(in_1k_root, in_sal_root, split):
    with open(os.path.join(in_1k_root, "labels.txt"), "r") as f:
        lines = f.readlines()

    classes = []
    for line in lines:
        line = line.strip()
        wn_id = line.split(",")[0]
        classes.append(wn_id)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    file_path_list, target_list = [], []

    for wn_id in tqdm(classes):
        cls_idx = class_to_idx[wn_id]
        in1k_cls_dir = os.path.join(in_1k_root, split, wn_id)
        file_name_list = os.listdir(in1k_cls_dir)

        in_shape_cls_dir = os.path.join(in_sal_root, split, wn_id)

        for filename in file_name_list:
            filename_no_ext = os.path.splitext(filename)[0]

            bg_file_path = os.path.join(
                in_shape_cls_dir, f"{filename_no_ext}_bg_.png"
            )
            fg_file_path = os.path.join(
                in_shape_cls_dir, f"{filename_no_ext}_fg_.png"
            )
            mask_file_path = os.path.join(
                in_shape_cls_dir, f"{filename_no_ext}_fg_mask_.png"
            )

            if not os.path.exists(bg_file_path):
                continue
            if not os.path.exists(fg_file_path):
                continue
            if not os.path.exists(mask_file_path):
                continue

            rel_file_path = os.path.join(wn_id, filename_no_ext)
            file_path_list.append(rel_file_path)
            target_list.append(cls_idx)

    output_file_path = os.path.join(in_sal_root, f"{split}_meta_data.txt")
    with open(output_file_path, "w") as f:
        for file_path, target in zip(file_path_list, target_list):
            f.write(f"{file_path}, {target}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_1k_root", type=str, default="data/imagenet")
    parser.add_argument("--in_sal_root", type=str, default="data/imagenet_sal")
    args = parser.parse_args()

    build_imagenet_sal_meta_data(args.in_1k_root, args.in_sal_root, "train")


if __name__ == "__main__":
    main()
