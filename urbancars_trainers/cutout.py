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

# refer to

import torch
import torchvision.transforms as transforms

from .erm import ERMTrainer


class CutoutTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "cutout"

        default_name = (
            f"{args.method}"
            f"_p_{args.cutout}"
            f"_es_{args.early_stop_metric}_{args.dataset}"
        )
        self.default_name = default_name

    def _get_train_transform(self):
        args = self.args
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
            transforms.RandomErasing(p=args.cutout)
        ]

        transform = transforms.Compose(trans)
        return transform
