"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from AugMix:
# https://github.com/google-research/augmix
# --------------------------------------------------------

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from model.augmix.augmix_transforms import AugMixDataset
from torchvision import transforms


class AugMixTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "augmix"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _get_train_transform(self):
        crop_size = self._get_resize_and_crop_size()[1]
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
            ]
        )
        return train_transform

    def _modify_train_set(self, train_dataset):
        mean, std = self._get_normalize_mean_std()
        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        modified_train_dataset = AugMixDataset(train_dataset, preprocess)
        return modified_train_dataset

    def _setup_criterion(self):
        super()._setup_criterion()
        self.kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["target"]
            aug1, aug2 = data_dict["aug1"], data_dict["aug2"]
            image_all = torch.cat([image, aug1, aug2], dim=0)
            image_all = image_all.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(image_all)
                logits_all = self.classifier(feature)
                logits_clean, logits_aug1, logits_aug2 = torch.split(
                    logits_all, image.size(0)
                )

                # Cross-entropy is only computed on clean images
                loss = self.criterion(logits_clean, target)
                p_clean, p_aug1, p_aug2 = (
                    torch.softmax(logits_clean, dim=1),
                    torch.softmax(logits_aug1, dim=1),
                    torch.softmax(logits_aug2, dim=1),
                )
                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp(
                    (p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1
                ).log()
                loss += (
                    12
                    * (
                        self.kl_criterion(p_mixture, p_clean)
                        + self.kl_criterion(p_mixture, p_aug1)
                        + self.kl_criterion(p_mixture, p_aug2)
                    )
                    / 3.0
                )

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
