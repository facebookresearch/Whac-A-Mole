"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from LfF:
# https://github.com/alinlab/LfF
# --------------------------------------------------------

import torch
import torch.nn as nn


from utils import IdxDataset, EMAGPU as EMA
from tqdm import tqdm
from model.criterion import GeneralizedCECriterion
from .base_trainer import BaseTrainer


class LfFTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "lff"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _method_specific_setups(self):
        train_target_attr = self.train_set.targets
        self.sample_loss_ema_b = EMA(
            torch.LongTensor(train_target_attr).to(self.device),
            alpha=0.7,
            device=self.device,
        )
        self.sample_loss_ema_d = EMA(
            torch.LongTensor(train_target_attr).to(self.device),
            alpha=0.7,
            device=self.device,
        )

    def _modify_train_set(self, train_dataset):
        return IdxDataset(train_dataset)

    def _setup_models(self):
        super(LfFTrainer, self)._setup_models()
        linear_weight_shape = self.linear_weight_shape
        self.bias_discover_net = nn.Linear(
            linear_weight_shape[1], linear_weight_shape[0]
        ).to(self.device)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCECriterion()

    def _setup_optimizers(self):
        super(LfFTrainer, self)._setup_optimizers()
        args = self.args
        self.optimizer_bias_discover_net = torch.optim.SGD(
            self.bias_discover_net.parameters(),
            self.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    def train(self):
        self.adjust_learning_rate(
            self.optimizer_bias_discover_net, self.init_lr, self.cur_epoch - 1
        )

        args = self.args
        self.bias_discover_net.train()
        self.classifier.train()

        total_cls_loss = 0
        total_ce_loss = 0
        total_gce_loss = 0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for idx, (idx_data, data_dict) in enumerate(pbar):
            img, label = data_dict["image"], data_dict["target"]
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(img)

                spurious_logits = self.bias_discover_net(feature)
                target_logits = self.classifier(feature)
                ce_loss = self.criterion(target_logits, label)
                gce_loss = self.gce_criterion(spurious_logits, label).mean()

            loss_b = self.criterion(spurious_logits, label).detach()
            loss_d = ce_loss.detach()
            idx_data = idx_data.to(self.device, non_blocking=True)

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, idx_data)
            self.sample_loss_ema_d.update(loss_d, idx_data)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[idx_data].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[idx_data].clone().detach()

            max_loss_b = self.sample_loss_ema_b.max_loss(label)
            max_loss_d = self.sample_loss_ema_d.max_loss(label)
            loss_b /= max_loss_b
            loss_d /= max_loss_d

            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            ce_loss = (ce_loss * loss_weight).mean()

            loss = ce_loss + gce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_bias_discover_net.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._optimizer_step(self.optimizer_bias_discover_net)

            self._scaler_update()

            total_cls_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_gce_loss += gce_loss.item()
            avg_cls_loss = total_cls_loss / (idx + 1)
            avg_ce_loss = total_ce_loss / (idx + 1)
            avg_gce_loss = total_gce_loss / (idx + 1)

            pbar.set_description(
                "[{}/{}] cls_loss: {:.3f}, ce: {:.3f}, gce: {:.3f}".format(
                    self.cur_epoch,
                    args.epoch,
                    avg_cls_loss,
                    avg_ce_loss,
                    avg_gce_loss,
                )
            )

        log_dict = {
            "loss": total_cls_loss / len(self.train_loader),
            "ce_loss": total_ce_loss / len(self.train_loader),
            "gce_loss": total_gce_loss / len(self.train_loader),
        }
        self.log_to_wandb(log_dict)

    def _state_dict_for_save(self):
        state_dict = super(LfFTrainer, self)._state_dict_for_save()
        state_dict.update(
            {
                "bias_discover_net": self.bias_discover_net.state_dict(),
                "optimizer_bias_discover_net": self.optimizer_bias_discover_net.state_dict(),
            }
        )
        return state_dict

    def _load_state_dict(self, state_dict):
        super(LfFTrainer, self)._load_state_dict(state_dict)
        self.bias_discover_net.load_state_dict(state_dict["bias_discover_net"])
        self.optimizer_bias_discover_net.load_state_dict(
            state_dict["optimizer_bias_discover_net"]
        )
