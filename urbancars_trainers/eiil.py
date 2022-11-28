"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from EIIL:
# https://github.com/ecreager/eiil
# --------------------------------------------------------

import torch
import torch.nn as nn


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from torch.utils.data.sampler import WeightedRandomSampler
from torch import autograd


class EIILTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "eiil"
        default_name = f"{args.method}_T_{args.bias_id_epoch}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def infer_group_labels(self):
        # train ERM model with T epochs
        args = self.args
        self.classifier.train()

        erm_id_optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        for idx_epoch in range(args.bias_id_epoch):
            losses = AverageMeter("Loss", ":.4e")

            pbar = tqdm(self.train_loader, dynamic_ncols=True)
            for data_dict in pbar:
                image, target = data_dict["image"], data_dict["label"]
                obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
                image = image.to(self.device, non_blocking=True)
                obj_gt = obj_gt.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = self.classifier(image)
                    loss = self.criterion(output, obj_gt).mean()

                self._loss_backward(loss)
                self._optimizer_step(erm_id_optimizer)
                self._scaler_update()
                erm_id_optimizer.zero_grad(set_to_none=True)

                losses.update(loss.item(), image.size(0))

                pbar.set_description(
                    f"[{idx_epoch}/{args.bias_id_epoch}] loss: {losses.avg:.4f}"
                )

        # infer loss
        self.classifier.eval()
        scale = torch.tensor(1.0, device=self.device, requires_grad=True)

        logits_lst = []
        label_list = []

        args = self.args
        ordered_train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

        pbar = tqdm(
            ordered_train_loader, dynamic_ncols=True, desc="inference for loss"
        )
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            image = image.to(self.device)
            target = target.to(self.device)
            label = target[:, 0]
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    logits = self.classifier(image)
                logits_lst.append(logits)
                label_list.append(label)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits_lst = torch.cat(logits_lst, dim=0)
            label_list = torch.cat(label_list, dim=0)
            loss = self.criterion(logits_lst * scale, label_list)

        # learning soft environment assignments
        env_w = torch.randn(
            len(loss), device=self.device, requires_grad=True
        )

        optimizer = torch.optim.Adam([env_w], lr=0.001)
        pbar = tqdm(range(self.args.eiil_n_steps), dynamic_ncols=True)
        for _ in pbar:
            with torch.cuda.amp.autocast(enabled=args.amp):
                total_n_penalty = 0
                for idx_class in range(self.num_class):
                    class_mask = label_list == idx_class

                    loss_cur_class = loss[class_mask]
                    env_w_cur_class = env_w[class_mask]

                    # penalty for env a
                    lossa = (
                        loss_cur_class * env_w_cur_class.sigmoid()
                    ).mean()
                    grada = autograd.grad(
                        lossa, [scale], create_graph=True
                    )[0]
                    penaltya = torch.sum(grada**2)
                    # penalty for env b
                    lossb = (
                        loss_cur_class * (1 - env_w_cur_class.sigmoid())
                    ).mean()
                    gradb = autograd.grad(
                        lossb, [scale], create_graph=True
                    )[0]
                    penaltyb = torch.sum(gradb**2)
                    penalty_list = [penaltya, penaltyb]

                    # negate
                    npenalty = -torch.stack(penalty_list).mean()
                    total_n_penalty += npenalty

            self._loss_backward(total_n_penalty, retain_graph=True)
            self._optimizer_step(optimizer)
            self._scaler_update()
            optimizer.zero_grad(set_to_none=True)

            total_n_penalty = total_n_penalty.item()
            desc = f"optimize environment: {total_n_penalty:.4f}"
            pbar.set_description(desc)

        # split envs
        hard_envs = (env_w.sigmoid() > 0.5).long()

        self.hard_envs = hard_envs

        train_set = self.train_set
        train_set.set_num_group_and_group_array(
            2, hard_envs.to("cpu")
        )
        weights = train_set.get_sampling_weights()
        sampler = WeightedRandomSampler(
            weights, len(train_set), replacement=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampler=sampler,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = (
            train_loader  # new train loader for training with groupdro
        )

        # reinitialize the classifier and optimizer
        self._setup_models()
        self._setup_optimizers()

    def _setup_for_groupdro(self):
        num_group = self.train_set.num_group
        self.num_group = num_group
        self.adv_probs = torch.ones(num_group, device=self.device) / num_group
        self.group_range = torch.arange(
            num_group, dtype=torch.long, device=self.device
        ).unsqueeze(1)

    def _before_train(self):
        self.infer_group_labels()
        self._setup_for_groupdro()

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            group_index = data_dict["group_index"].to(
                self.device, non_blocking=True
            )

            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)
                loss_per_sample = self.criterion(output, obj_gt)

                # compute group loss
                group_map = (group_index == self.group_range).float()
                group_count = group_map.sum(1)
                group_denom = (
                    group_count + (group_count == 0).float()
                )  # avoid nans
                group_loss = (
                    group_map @ loss_per_sample.flatten()
                ) / group_denom

                # update adv_probs
                with torch.no_grad():
                    self.adv_probs = self.adv_probs * torch.exp(
                        args.groupdro_robust_step_size * group_loss.detach()
                    )
                    self.adv_probs = self.adv_probs / (self.adv_probs.sum())

                # compute reweighted robust loss
                loss = group_loss @ self.adv_probs

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
        state_dict["adv_probs"] = self.adv_probs
        state_dict["hard_envs"] = self.hard_envs
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
        self.adv_probs = state_dict["adv_probs"]
        self.hard_envs = state_dict["hard_envs"]

        self._setup_for_groupdro()
