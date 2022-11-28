"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from TorchVision:
# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# --------------------------------------------------------

import torchvision.transforms as transforms


from .erm import ERMTrainer


class CutoutTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "cutout"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_cutout_{args.cutout}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _get_train_transform(self):
        args = self.args
        mean, std = self._get_normalize_mean_std()
        normalize = transforms.Normalize(mean=mean, std=std)
        crop_size = self._get_resize_and_crop_size()[1]
        trans = [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=args.cutout)
        ]

        train_transform = transforms.Compose(trans)
        return train_transform
