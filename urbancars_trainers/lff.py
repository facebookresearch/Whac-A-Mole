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
from model.classifiers import get_classifier
from .base_trainer import BaseTrainer


class LfFTrainer(BaseTrainer):
    def _method_specific_setups(self):
        train_target_attr = self.train_set.get_labels()[:, 0]
        self.sample_loss_ema_b = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )
        self.sample_loss_ema_d = EMA(
            torch.LongTensor(train_target_attr), device=self.device, alpha=0.7
        )

    def _modify_train_set(self, train_dataset):
        return IdxDataset(train_dataset)

    def _setup_models(self):
        super(LfFTrainer, self)._setup_models()
        self.bias_discover_net = get_classifier(
            arch=self.args.arch,
            num_classes=self.num_class,
        ).to(self.device)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.gce_criterion = GeneralizedCECriterion()

    def _setup_optimizers(self):
        super(LfFTrainer, self)._setup_optimizers()
        args = self.args
        if args.optimizer == "sgd":
            self.optimizer_bias_discover_net = torch.optim.SGD(
                self.bias_discover_net.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "lff"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def train(self):
        args = self.args
        self.bias_discover_net.train()
        self.classifier.train()

        total_cls_loss = 0
        total_ce_loss = 0
        total_gce_loss = 0

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for idx, (idx_data, data_dict) in enumerate(pbar):
            img, all_attr_label = data_dict["image"], data_dict["label"]
            img = img.to(self.device, non_blocking=True)
            label = all_attr_label[:, 0]
            label = label.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                spurious_logits = self.bias_discover_net(img)
                target_logits = self.classifier(img)
                ce_loss = self.criterion(target_logits, label)
                gce_loss = self.gce_criterion(spurious_logits, label).mean()

            loss_b = self.criterion(spurious_logits, label).detach()
            loss_d = ce_loss.detach()

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
