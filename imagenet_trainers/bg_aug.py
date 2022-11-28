"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.transforms as transforms


from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils import AverageMeter
from model.transforms import CoRandomResizedCrop, CoRandomHorizontalFlip
from dataset.imagenet_sal import ImageNetSal, imagenet_sal_collate


class BackgroundAugTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "bg_aug"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )

        self.default_name = default_name

    def _modify_train_set(self, train_dataset):
        crop_size = self._get_resize_and_crop_size()[1]
        mean, std = self._get_normalize_mean_std()

        bg_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        co_transform = transforms.Compose(
            [
                CoRandomResizedCrop(crop_size),
                CoRandomHorizontalFlip(),
            ]
        )
        final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        mask_final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        mixed_rand_dataset = ImageNetSal(
            root=args.data_root,
            split="train",
            co_transform=co_transform,
            final_transform=final_transform,
            mask_final_transform=mask_final_transform,
            bg_transform=bg_transform,
        )

        self.len_in1k_dataset = len(train_dataset)
        self.len_mixed_rand = len(mixed_rand_dataset)
        concat_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, mixed_rand_dataset]
        )
        return concat_dataset

    def _get_train_loader(self, train_set):
        args = self.args

        in1k_weights = [1] * self.len_in1k_dataset
        mixed_rand_weights = [1] * self.len_mixed_rand

        weights = in1k_weights + mixed_rand_weights
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, self.len_in1k_dataset, replacement=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            sampler=sampler,
            collate_fn=imagenet_sal_collate,
        )
        return train_loader

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            concat_image_list = []
            concat_target_list = []

            if "image" in data_dict:
                image, target = data_dict["image"], data_dict["target"]
                image = image.to(self.device, non_blocking=True)

                concat_image_list.append(image)
                concat_target_list.append(target)

            if "bg_aug_target" in data_dict:
                fg_image = data_dict["fg_image"].to(self.device, non_blocking=True)
                bg_image = data_dict["bg_image"].to(self.device, non_blocking=True)
                mask = data_dict["mask"].to(self.device, non_blocking=True)
                bg_aug_target = data_dict["bg_aug_target"]
                mixed_image = fg_image * mask + bg_image * (1 - mask)
                concat_image_list.append(mixed_image)
                concat_target_list.append(bg_aug_target)

            image = torch.cat(concat_image_list, dim=0)
            target = torch.cat(concat_target_list, dim=0).to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(image)
                output = self.classifier(feature)
                loss = self.criterion(output, target)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )

        self.log_to_wandb({"loss": losses.avg})
