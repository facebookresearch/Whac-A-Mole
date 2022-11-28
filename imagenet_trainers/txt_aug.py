"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


from .erm import ERMTrainer
from dataset.imagenet_stylized import ImageNetStylized


class TextureAugTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "txt_aug"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )

        self.default_name = default_name

    def _modify_train_set(self, train_dataset):
        aug_dataset = ImageNetStylized(
            root=args.data_root,
            split="train",
            transform=self._get_train_transform(),
            return_dist_shift_index=False,
        )

        self.len_in1k_dataset = len(train_dataset)
        self.len_stylized = len(aug_dataset)

        concat_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, aug_dataset]
        )
        return concat_dataset

    def _get_train_loader(self, train_set):
        args = self.args

        in1k_weights = [1] * self.len_in1k_dataset
        mixed_rand_weights = [1] * self.len_stylized

        weights = in1k_weights + mixed_rand_weights
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, self.len_in1k_dataset, replacement=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            sampler=sampler,
        )
        return train_loader
