"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Gradient Starvation:
# https://github.com/mohammadpz/Gradient_Starvation
# --------------------------------------------------------

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class SpectralDecoupleTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "sd"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _setup_optimizers(self):
        args = self.args
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=0.0,  # no weight decay for spectral decouple
            )
        else:
            raise NotImplementedError

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = self.classifier(image)
                ce_loss = self.criterion(logits, obj_gt)
                logits_norm = ((logits[range(logits.shape[0]), obj_gt]) ** 2).mean()
                loss = ce_loss + args.sp * logits_norm

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
