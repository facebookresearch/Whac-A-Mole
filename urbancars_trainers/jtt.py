"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from JTT:
# https://github.com/anniesch/jtt
# --------------------------------------------------------

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from torch.utils.data import Subset, ConcatDataset


class JTTTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "jtt"
        default_name = f"{args.method}_T_{args.bias_id_epoch}_up_{args.jtt_up_weight}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        args = self.args
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        return train_loader

    def infer_error_set(self):
        # train ERM model with T epoch
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        erm_id_optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        for idx_jtt_epoch in range(args.bias_id_epoch):
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
                self._optimizer_step(erm_id_optimizer)
                self._scaler_update()
                erm_id_optimizer.zero_grad(set_to_none=True)

                losses.update(loss.item(), image.size(0))

                pbar.set_description(
                    f"[{idx_jtt_epoch}/{args.bias_id_epoch}] loss:"
                    f" {losses.avg:.4f}"
                )

        # infer loss
        self.classifier.eval()

        error_set_list = []

        ordered_train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            shuffle=False,  # no shuffle for inferring error set
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

        pbar = tqdm(
            ordered_train_loader,
            dynamic_ncols=True,
            desc="inference for prediction",
        )
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            image = image.to(self.device)
            target = target.to(self.device)
            label = target[:, 0]
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    logits = self.classifier(image)
                pred = logits.argmax(dim=1)
                error = pred != label
                error_set_list.append(error.long().cpu())

        error_set_list = torch.cat(error_set_list, dim=0)
        error_indices = torch.nonzero(error_set_list).flatten().tolist()

        train_set = self.train_set
        upsampled_points = Subset(train_set, error_indices * args.jtt_up_weight)
        concat_train_set = ConcatDataset([train_set, upsampled_points])
        train_loader = torch.utils.data.DataLoader(
            concat_train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = train_loader

        # reinitialize the classifier and optimizer
        self._setup_models()
        self._setup_optimizers()

    def _before_train(self):
        self.infer_error_set()

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

    def _state_dict_for_save(self):
        state_dict = super()._state_dict_for_save()
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
