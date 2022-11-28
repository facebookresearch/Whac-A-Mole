"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from DFR:
# https://github.com/PolinaKirichenko/deep_feature_reweighting
# --------------------------------------------------------

import torch

from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class DFRTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "dfr"
        default_name = (
            f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        )
        self.default_name = default_name

    def _before_train(self):
        args = self.args
        best_state_dict = None
        for _ in range(self.cur_epoch, args.epoch + 1):
            self.train_erm()
            is_best = self.eval()
            if is_best:
                best_state_dict = self._state_dict_for_save()
            self.cur_epoch += 1

        self._load_state_dict(best_state_dict)
        self.classifier.fc = torch.nn.Linear(
            self.classifier.fc.in_features, self.num_class
        ).to(self.device)  # reinitialize fc layer

        for p in self.classifier.parameters():
            p.requires_grad = False

        for p in self.classifier.fc.parameters():
            p.requires_grad = True

        self._setup_optimizers()

        train_set = self.train_set
        indices = train_set._get_subsample_group_indices(args.group_label)
        train_set = torch.utils.data.Subset(train_set, indices)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.cond_best_acc = 0
        self.cond_on_best_val_log_dict = {}
        self.cur_epoch = 1

    def _set_train(self):
        self.classifier.eval()  # freeze BN for linear probing

    def train_erm(self):
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
                output = self.classifier(image)
                loss = self.criterion(output, obj_gt)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"ERM: [{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

    def train(self):
        args = self.args
        self._set_train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
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
