"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class ERMTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "erm"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["target"]
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(image)
                output = self.classifier(feature)
                loss = self.criterion(output, target)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
