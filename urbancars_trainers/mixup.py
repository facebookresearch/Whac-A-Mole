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


from model.mixup_cutmix_transforms import RandomMixup
from torch.utils.data.dataloader import default_collate
from .base_trainer import BaseTrainer
from utils import AverageMeter
from tqdm import tqdm


class MixupTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "mixup"

        default_name = (
            f"{args.method}"
            f"_alpha_{args.mixup_alpha}"
            f"_es_{args.early_stop_metric}_{args.dataset}"
        )
        self.default_name = default_name

    def _get_train_collate_fn(self):
        args = self.args
        collate_fn = None
        num_classes = self.num_class
        mixup = RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)

        def collate_fn(batch):
            collate_batch = default_collate(batch)
            batch_image = collate_batch["image"]
            batch_label = collate_batch["label"][:, 0]
            return mixup(batch_image, batch_label)

        return collate_fn

    def train(self):
        args = self.args
        self._set_train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for i, data_dict in enumerate(pbar):
            image, obj_gt = data_dict["image"], data_dict["label"]
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)
                loss = self.criterion(output, obj_gt)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
