"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from DebiAN:
# https://github.com/zhihengli-UR/DebiAN
# --------------------------------------------------------

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from model.classifiers import get_classifier
from utils import EPS


class DebiANTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "debian"

        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _method_specific_setups(self):
        self.second_train_loader = self._get_train_loader(self.train_set)

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _setup_models(self):
        super()._setup_models()
        args = self.args
        self.bias_discover_net = get_classifier(
            args.arch,
            self.num_class,
        ).to(self.device)

    def _setup_optimizers(self):
        super()._setup_optimizers()
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

    def _train_classifier(self, data_dict):
        args = self.args
        self.classifier.train()
        self.bias_discover_net.eval()

        image, target = data_dict["image"], data_dict["label"]
        obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj

        image = image.to(self.device, non_blocking=True)
        obj_gt = obj_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            spurious_logits = self.bias_discover_net(image)
        with torch.cuda.amp.autocast(enabled=args.amp):
            target_logits = self.classifier(image)

            label = obj_gt.long()
            label = label.reshape(target_logits.shape[0])

            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            ce_loss = self.criterion(target_logits, label)

            # reweight CE with DEO
            for target_val in range(self.num_class):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                    p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                    (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                if (
                    negative_spurious_group_avg_p
                    < positive_spurious_group_avg_p
                ):
                    p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val

                weight = 1 + p_spurious_w_same_t_val
                ce_loss[batch_bool] *= weight

            ce_loss = ce_loss.mean()

        self._loss_backward(ce_loss)
        self._optimizer_step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)

        return ce_loss.item()

    def _train_bias_discover_net(self, data_dict):
        args = self.args
        self.bias_discover_net.train()
        self.classifier.eval()

        image, target = data_dict["image"], data_dict["label"]
        obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj

        image = image.to(self.device, non_blocking=True)
        obj_gt = obj_gt.to(self.device, non_blocking=True)

        with torch.no_grad():
            target_logits = self.classifier(image)

        with torch.cuda.amp.autocast(enabled=args.amp):
            spurious_logits = self.bias_discover_net(image)
            label = obj_gt.long()
            label = label.reshape(target_logits.shape[0])
            p_vanilla = torch.softmax(target_logits, dim=1)
            p_spurious = torch.sigmoid(spurious_logits)

            # ==== deo loss ======
            sum_discover_net_deo_loss = 0
            sum_penalty = 0
            num_classes_in_batch = 0
            for target_val in range(self.num_class):
                batch_bool = label.long().flatten() == target_val
                if not batch_bool.any():
                    continue
                p_vanilla_w_same_t_val = p_vanilla[batch_bool, target_val]
                p_spurious_w_same_t_val = p_spurious[batch_bool, target_val]

                positive_spurious_group_avg_p = (
                    p_spurious_w_same_t_val * p_vanilla_w_same_t_val
                ).sum() / (p_spurious_w_same_t_val.sum() + EPS)
                negative_spurious_group_avg_p = (
                    (1 - p_spurious_w_same_t_val) * p_vanilla_w_same_t_val
                ).sum() / ((1 - p_spurious_w_same_t_val).sum() + EPS)

                discover_net_deo_loss = -torch.log(
                    EPS
                    + torch.abs(
                        positive_spurious_group_avg_p
                        - negative_spurious_group_avg_p
                    )
                )

                negative_p_spurious_w_same_t_val = 1 - p_spurious_w_same_t_val
                penalty = -torch.log(
                    EPS
                    + 1
                    - torch.abs(
                        p_spurious_w_same_t_val.mean()
                        - negative_p_spurious_w_same_t_val.mean()
                    )
                )

                sum_discover_net_deo_loss += discover_net_deo_loss
                sum_penalty += penalty
                num_classes_in_batch += 1

            sum_penalty /= num_classes_in_batch
            sum_discover_net_deo_loss /= num_classes_in_batch
            loss_discover = sum_discover_net_deo_loss + sum_penalty

        self._loss_backward(loss_discover)
        self._optimizer_step(self.optimizer_bias_discover_net)
        self.optimizer_bias_discover_net.zero_grad(set_to_none=True)

        return loss_discover.item()

    def train(self):
        args = self.args
        cls_losses = AverageMeter("cls_loss", ":.4e")
        dis_losses = AverageMeter("dis_loss", ":.4e")

        pbar = tqdm(
            zip(self.train_loader, self.second_train_loader),
            dynamic_ncols=True,
            total=len(self.train_loader),
        )

        for main_data_dict, second_data_dict in pbar:
            cls_loss = self._train_classifier(main_data_dict)
            dis_loss = self._train_bias_discover_net(second_data_dict)

            cls_losses.update(cls_loss, main_data_dict["image"].size(0))
            dis_losses.update(dis_loss, main_data_dict["image"].size(0))

            self._scaler_update()

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] cls_loss:"
                f" {cls_losses.avg:.4f} dis_loss: {dis_losses.avg:.4f}"
            )

        self.log_to_wandb(
            {"cls_loss": cls_losses.avg, "dis_loss": dis_losses.avg}
        )

    def _state_dict_for_save(self):
        state_dict = super()._state_dict_for_save()
        state_dict.update(
            {
                "bias_discover_net": self.bias_discover_net.state_dict(),
                "optimizer_bias_discover_net": self.optimizer_bias_discover_net.state_dict(),
            }
        )
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
        self.bias_discover_net.load_state_dict(state_dict["bias_discover_net"])
        self.optimizer_bias_discover_net.load_state_dict(
            state_dict["optimizer_bias_discover_net"]
        )
