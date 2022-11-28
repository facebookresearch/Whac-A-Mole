"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# Reference:
# https://github.com/zzzace2000/robust_cls_model
# --------------------------------------------------------

import torch
import torchvision.transforms as transforms


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter


class CFFAugTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "cf_f_aug"

        default_name = (
            f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        )
        self.default_name = default_name

    def _modify_train_set(self, train_dataset):
        obj_bbox_mask = torch.zeros(1, 1, 256, 256, device=self.device)
        img_size = 256
        start = int(img_size * 0.25)
        bbox_len = int(img_size * 0.5)
        obj_bbox_mask[
            :, :, start : start + bbox_len, start : start + bbox_len
        ] = 1

        self.obj_bbox_mask = obj_bbox_mask
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.normalize_transform = normalize
        return train_dataset

    def train(self):
        args = self.args
        self._set_train()
        losses = AverageMeter("Loss", ":.4e")
        factual_losses = AverageMeter("FactualLoss", ":.4e")
        cf_losses = AverageMeter("CFLoss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]

            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)

            b, c, h, w = image.shape

            with torch.no_grad():
                random_bg = image.new(b, c, 1, 1).uniform_().repeat(1, 1, h, w)
                random_bg += image.new(*image.size()).normal_(0, 0.2)
                random_bg.clamp_(0.0, 1.0)
                random_bg = self.normalize_transform(random_bg)

                factual_img = image * self.obj_bbox_mask + random_bg * (
                    1 - self.obj_bbox_mask
                )
                cf_img = (1 - self.obj_bbox_mask) * image

            concat_img = torch.cat([image, factual_img, cf_img], dim=0)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(concat_img)
                output, factual_output, cf_output = torch.split(
                    output, b, dim=0
                )
                ce_loss = self.criterion(output, obj_gt)
                factual_loss = self.criterion(factual_output, obj_gt)
                cf_loss = self.criterion(cf_output, 1 - obj_gt)

                loss = ce_loss + factual_loss + cf_loss

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(ce_loss.item(), image.size(0))
            factual_losses.update(factual_loss.item(), image.size(0))
            cf_losses.update(cf_loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f},"
                f" f: {factual_losses.avg:.4f}, cf: {cf_losses.avg:.4f}"
            )

        self.log_to_wandb(
            {
                "loss": losses.avg,
                "factual_loss": factual_losses.avg,
                "cf_loss": cf_losses.avg,
            }
        )
