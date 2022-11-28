"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from TorchVision:
# https://github.com/pytorch/vision/blob/main/references/classification/train.py
# --------------------------------------------------------


import torch

from .erm import ERMTrainer
from model.mixup_cutmix_transforms import RandomMixup
from torch.utils.data.dataloader import default_collate


class MixupTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "mixup"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_alpha_{args.mixup_alpha}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        args = self.args
        num_classes = self.num_class
        random_mixup = RandomMixup(
            num_classes, p=1.0, alpha=args.mixup_alpha, target_key="target"
        )

        def collate_fn(batch):
            collate_batch = default_collate(batch)
            batch_image = collate_batch["image"]
            batch_label = collate_batch["target"]
            return random_mixup(batch_image, batch_label)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_fn,
        )
        return train_loader
