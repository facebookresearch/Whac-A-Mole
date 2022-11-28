"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from SUBG:
# https://github.com/facebookresearch/BalancingGroups
# --------------------------------------------------------

import torch

from .erm import ERMTrainer


class SUBGTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "subg"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        args = self.args
        indices = train_set._get_subsample_group_indices(args.group_label)
        train_set = torch.utils.data.Subset(train_set, indices)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        return train_loader
