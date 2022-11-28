"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import json
import glob
import copy
import random
import numpy as np
import torch


from create_datasets.urbancars.src_dataset.lvis import LVISDataset
from create_datasets.urbancars.src_dataset.places import Places
from data_utils.object_scale import (
    crop_mask_and_img,
    rescale_cropped_mask_and_img,
)
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset
from create_datasets.urbancars.src_dataset.stanford_cars import StanfordCars
from logging import warning


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class UrbanCarsDataGen:
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args

        with open(args.meta_data_fpath) as f:
            meta_data = json.load(f)

        self.obj_dict_list = meta_data["object"]
        self.bg_dict_list = meta_data["background"]
        self.co_occur_obj_dict_list = meta_data["co_occur_object"]

        assert (
            len(self.obj_dict_list)
            == len(self.bg_dict_list)
            == len(self.co_occur_obj_dict_list)
        )
        self.num_composition = len(self.obj_dict_list) ** 3

        # build image loaders for foreground and background images

        indices = list(range(args.num_train, args.num_train + args.num_val))
        random.Random(0).shuffle(indices)

        if args.split == "train":
            split_indices = list(range(args.num_train))
            num_img_split = args.num_train
        elif args.split == "val":
            split_indices = list(
                range(args.num_train, args.num_train + args.num_val)
            )
            num_img_split = args.num_val
        elif args.split == "test":
            split_indices = list(
                range(
                    args.num_train + args.num_val,
                    args.num_train + args.num_val + args.num_test,
                )
            )
            num_img_split = args.num_test
        else:
            raise NotImplementedError

        self.num_img_split = num_img_split

        self.obj_cat_name_to_iter_loader = {}
        for obj_dict in tqdm(
            self.obj_dict_list, dynamic_ncols=True, desc="loading obj"
        ):
            obj_name = obj_dict["name"]
            obj_id = obj_dict["id"]
            stanford_cars_train = StanfordCars(
                "data",
                obj_id,
                split="train",
                transform=ToTensor(),
                mask_transform=ToTensor(),
            )
            stanford_cars_test = StanfordCars(
                "data",
                obj_id,
                split="test",
                transform=ToTensor(),
                mask_transform=ToTensor(),
            )
            stanford_cars = ConcatDataset(
                [stanford_cars_train, stanford_cars_test]
            )
            assert (
                len(stanford_cars)
                >= args.num_train + args.num_val + args.num_test
            )
            subsample_indices = list(range(len(stanford_cars)))
            random.Random(0).shuffle(subsample_indices)
            stanford_cars = Subset(stanford_cars, subsample_indices)
            stanford_cars = Subset(stanford_cars, split_indices)
            loader = torch.utils.data.DataLoader(
                stanford_cars,
                batch_size=1,
            )
            self.obj_cat_name_to_iter_loader[obj_name] = iter(loader)

        self.bg_cat_name_to_iter_loader = {}
        for bg_dict in self.bg_dict_list:
            bg_cat_name = bg_dict["name"]

            places_dataset_list = []

            for places_dict in bg_dict["class_list"]:
                places_bg_name = places_dict["name"]
                places_dataset = Places(
                    "data",
                    places_bg_name,
                    ToTensor(),
                )
                places_dataset_list.append(places_dataset)

            concat_places_dataset = ConcatDataset(places_dataset_list)
            assert (
                len(concat_places_dataset)
                >= args.num_train + args.num_val + args.num_test
            )

            subsample_indices = list(range(len(concat_places_dataset)))
            random.Random(0).shuffle(subsample_indices)
            concat_places_dataset = Subset(
                concat_places_dataset, subsample_indices
            )
            concat_places_dataset = Subset(concat_places_dataset, split_indices)
            loader = torch.utils.data.DataLoader(
                concat_places_dataset,
                batch_size=1,
            )
            self.bg_cat_name_to_iter_loader[bg_cat_name] = iter(loader)

        self.co_occur_obj_cat_name_to_iter_loader = {}
        for co_occur_obj_dict in tqdm(
            self.co_occur_obj_dict_list,
            dynamic_ncols=True,
            desc="loading co_occur_obj",
        ):
            co_occur_obj_name = co_occur_obj_dict["name"]
            co_occur_obj_lvis_dataset_list = []

            if co_occur_obj_name == "urban":
                area_thresh = 4500
                threshold_standard = "bbox"
            elif co_occur_obj_name == "country":
                area_thresh = 3500
                threshold_standard = "mask"
            else:
                raise NotImplementedError

            for lvis_obj_dict in co_occur_obj_dict["class_list"]:
                co_occur_obj_cat_id = lvis_obj_dict["id"]
                co_occur_obj_lvis_dataset = LVISDataset(
                    "data",
                    co_occur_obj_cat_id,
                    transform=ToTensor(),
                    area_thresh=area_thresh,
                    threshold_standard=threshold_standard,
                )
                co_occur_obj_lvis_dataset_list.append(co_occur_obj_lvis_dataset)

            concat_co_occur_obj_lvis_dataset = ConcatDataset(
                co_occur_obj_lvis_dataset_list
            )
            assert (
                len(concat_co_occur_obj_lvis_dataset)
                >= args.num_train + args.num_val + args.num_test
            )

            subsample_indices = list(
                range(len(concat_co_occur_obj_lvis_dataset))
            )
            random.Random(0).shuffle(subsample_indices)
            concat_co_occur_obj_lvis_dataset = Subset(
                concat_co_occur_obj_lvis_dataset, subsample_indices
            )
            concat_co_occur_obj_lvis_dataset = Subset(
                concat_co_occur_obj_lvis_dataset, split_indices
            )
            loader = torch.utils.data.DataLoader(
                concat_co_occur_obj_lvis_dataset,
                batch_size=1,
            )
            self.co_occur_obj_cat_name_to_iter_loader[co_occur_obj_name] = iter(
                loader
            )

        self.obj_scale = 0.5
        self.object_center_pos = (
            args.target_image_size // 2,
            args.target_image_size // 2,
        )
        self.co_occur_obj_scale = 0.25
        self.co_occur_obj_center_pos = (
            int(0.5 * args.target_image_size),
            int(0.875 * args.target_image_size),
        )

    def _gen_composition(
        self,
        factor_composition,
    ):
        args = self.args
        (
            idx_obj,
            idx_bg,
            idx_co_occur_obj,
        ) = factor_composition
        bg_align = idx_obj == idx_bg
        co_occur_obj_align = idx_obj == idx_co_occur_obj
        composition_bg_ratio = args.bg_ratio if bg_align else 1 - args.bg_ratio
        composition_co_occur_obj_ratio = (
            args.co_occur_obj_ratio
            if co_occur_obj_align
            else 1 - args.co_occur_obj_ratio
        )
        composition_ratio = (
            composition_bg_ratio * composition_co_occur_obj_ratio
        )

        num_img = int(composition_ratio * self.num_img_split)
        if num_img == 0:
            num_img = 1
            warning("num_img is rounded up to 1")
        # assert num_img > 0

        obj_dict = self.obj_dict_list[idx_obj]
        bg_dict = self.bg_dict_list[idx_bg]
        co_occur_obj_dict = self.co_occur_obj_dict_list[idx_co_occur_obj]
        obj_name = obj_dict["name"]
        bg_name = bg_dict["name"]
        co_occur_obj_name = co_occur_obj_dict["name"]

        composition_dir_name = (
            f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
        )
        composition_dir_path = os.path.join(
            args.gen_data_dir, composition_dir_name
        )
        if not os.path.exists(composition_dir_path):
            os.makedirs(composition_dir_path, exist_ok=True)

        # check if the composition is already generated
        num_generated_img = len(
            glob.glob(os.path.join(composition_dir_path, "*.jpg"))
        )
        if num_generated_img >= num_img:
            return

        obj_loader = self.obj_cat_name_to_iter_loader[obj_name]
        bg_loader = self.bg_cat_name_to_iter_loader[bg_name]
        co_occur_obj_loader = self.co_occur_obj_cat_name_to_iter_loader[
            co_occur_obj_name
        ]

        for i in tqdm(range(num_img), leave=False, dynamic_ncols=True):
            obj_data_dict = next(obj_loader)
            image, mask = obj_data_dict["image"], obj_data_dict["mask"]
            image.squeeze_(0)
            mask.squeeze_(0)

            cropped_mask, cropped_image = crop_mask_and_img(mask, image)

            (rescaled_mask, rescaled_img,) = rescale_cropped_mask_and_img(
                cropped_mask=cropped_mask,
                cropped_img=cropped_image,
                scale=self.obj_scale,
                target_side_size=args.target_image_size,
                object_center_pos=self.object_center_pos,
            )

            bg_image = next(bg_loader)
            bg_image.squeeze_(0)

            co_occur_image, co_occur_mask = next(co_occur_obj_loader)
            co_occur_image.squeeze_(0)
            co_occur_mask.squeeze_(0)

            cropped_co_occur_mask, cropped_co_occur_image = crop_mask_and_img(
                co_occur_mask, co_occur_image
            )

            (
                rescaled_co_occur_mask,
                rescaled_co_occur_img,
            ) = rescale_cropped_mask_and_img(
                cropped_mask=cropped_co_occur_mask,
                cropped_img=cropped_co_occur_image,
                scale=self.co_occur_obj_scale,
                target_side_size=args.target_image_size,
                object_center_pos=self.co_occur_obj_center_pos,
            )

            blend_img = (
                rescaled_co_occur_mask * rescaled_co_occur_img
                + (1 - rescaled_co_occur_mask) * bg_image
            )

            output_img = (
                rescaled_mask * rescaled_img + (1 - rescaled_mask) * blend_img
            )

            co_occur_mask_fname = f"{i:03d}_co_occur_obj_mask.png"
            co_occur_mask_path = os.path.join(
                composition_dir_path, co_occur_mask_fname
            )
            save_image(
                rescaled_co_occur_mask.float(),
                co_occur_mask_path,
                nrow=1,
                padding=0,
                value_range=(0, 1),
            )

            img_fname = f"{i:03d}.jpg"
            img_fpath = os.path.join(composition_dir_path, img_fname)
            save_image(
                output_img, img_fpath, nrow=1, padding=0, value_range=(0, 1)
            )

            mask_fname = f"{i:03d}_mask.png"
            mask_fpath = os.path.join(composition_dir_path, mask_fname)
            save_image(
                rescaled_mask.float(),
                mask_fpath,
                nrow=1,
                padding=0,
                value_range=(0, 1),
            )

    def _composition_generator(self):
        for idx_obj in range(len(self.obj_dict_list)):
            for idx_bg in range(len(self.bg_dict_list)):
                for idx_co_occur_obj in range(len(self.co_occur_obj_dict_list)):
                    factor_composition = (
                        idx_obj,
                        idx_bg,
                        idx_co_occur_obj,
                    )
                    yield factor_composition

    def __call__(self):
        self.setup()
        composition_generator = self._composition_generator()
        for factor_composition in tqdm(
            composition_generator,
            total=self.num_composition,
            dynamic_ncols=True,
        ):
            self._gen_composition(factor_composition)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--meta_data_fpath",
        type=str,
        default="create_datasets/urbancars/metadata/urbancars/meta_data.json",
    )
    parser.add_argument(
        "--gen_data_dir", type=str, default="data/urbancars"
    )

    parser.add_argument("--num_train", type=int, default=4000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--target_image_size", type=int, default=256)
    parser.add_argument("--bg_ratio", type=float, default=0.95)
    parser.add_argument("--co_occur_obj_ratio", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=5)

    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    if not os.path.exists(args.gen_data_dir):
        os.makedirs(args.gen_data_dir)

    if args.split in ["val", "test"]:
        args.bg_ratio = 0.5
        args.co_occur_obj_ratio = 0.5

    bg_ratio = args.bg_ratio
    co_occur_obj_ratio = args.co_occur_obj_ratio
    corr_folder_name = (
        f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
    )

    gen_dir = os.path.join(
        args.gen_data_dir, corr_folder_name, args.split
    )
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    print("gen_data_dir: ", gen_dir)

    new_args = copy.deepcopy(args)
    new_args.gen_data_dir = gen_dir
    new_args.bg_ratio = bg_ratio
    new_args.co_occur_obj_ratio = co_occur_obj_ratio
    gen = UrbanCarsDataGen(new_args)
    gen()


if __name__ == "__main__":
    main()
