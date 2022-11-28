"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Domain Independent:
# https://github.com/princetonvisualai/DomainBiasMitigation
# --------------------------------------------------------

import torch


from .base_trainer import BaseTrainer
from model.classifiers import DomainIndependentClassifier
from utils import AverageMeter
from tqdm import tqdm


class DomainIndependentTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "di"

        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _setup_models(self):
        args = self.args
        if args.group_label in ["bg", "co_occur_obj"]:
            di_num_domain = 2
        else:
            assert args.group_label == "both"
            di_num_domain = 4
        self.classifier = DomainIndependentClassifier(
            args.arch, self.num_class, di_num_domain,
        ).to(self.device)

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            domain_label = data_dict["domain_label"]

            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)
            domain_label = domain_label.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits_per_domain = self.classifier(image)
                logits = logits_per_domain[
                    range(logits_per_domain.shape[0]), domain_label
                ]
                loss = self.criterion(logits, obj_gt)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
