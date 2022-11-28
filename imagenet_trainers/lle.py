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
from model.transforms import CoRandomResizedCrop, CoRandomHorizontalFlip, WatermarkAug
from dataset.imagenet_sal import (
    ImageNetSal,
    imagenet_sal_collate,
)
from dataset.imagenet_stylized import ImageNetStylized
from dataset.imagenet import ImageNet
from dataset.imagenet_edge import ImageNetEdge


class LLETrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "lle"
        edge_aug_str = "_edge_aug" if args.edge_aug else ""
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"{edge_aug_str}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )

        self.default_name = default_name

    def _modify_train_set(self, train_dataset):
        args = self.args

        crop_size = self._get_resize_and_crop_size()[1]
        mean, std = self._get_normalize_mean_std()

        # bg aug
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
            return_dist_shift=True,
        )

        # style aug
        stylized_imagenet = ImageNetStylized(
            root=args.data_root,
            split="train",
            transform=self._get_train_transform(),
            return_dist_shift_index=True,
        )

        # watermark aug
        normalize = transforms.Normalize(
            mean=mean, std=std
        )
        watermark_transform = WatermarkAug()
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                watermark_transform,
                transforms.ToTensor(),
                normalize,
            ]
        )
        watermark_aug_imagenet = ImageNet(
            args.data_root,
            "train",
            transform=train_transform,
            return_group_index=args.method == "eiil"
            and args.resume is not None,
            return_dist_shift_index=True,
            dist_shift_index=3,
        )
        aug_dataset_list = [mixed_rand_dataset, stylized_imagenet, watermark_aug_imagenet]

        other_dist_shift_index = 4

        if args.edge_aug:
            imagenet_edge = ImageNetEdge(
                args.data_root, self._get_train_transform(), return_dist_shift=True, dist_shift_index=other_dist_shift_index)
            aug_dataset_list.append(imagenet_edge)

        concat_aug_dataset = torch.utils.data.ConcatDataset(
            aug_dataset_list
        )

        self.len_in1k_dataset = len(train_dataset)
        self.len_aug_dataset = len(concat_aug_dataset)
        concat_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, concat_aug_dataset]
        )
        return concat_dataset

    def _get_train_loader(self, train_set):
        args = self.args

        in1k_weights = [1] * self.len_in1k_dataset
        aug_dataset_weights = [1] * self.len_aug_dataset

        weights = in1k_weights + aug_dataset_weights
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
        losses_domain = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            concat_image_list = []
            concat_target_list = []
            concat_dist_shift_list = []

            if "image" in data_dict:
                image, target, dist_shift = (
                    data_dict["image"],
                    data_dict["target"],
                    data_dict["dist_shift"],
                )
                image = image.to(self.device, non_blocking=True)

                concat_image_list.append(image)
                concat_target_list.append(target)
                concat_dist_shift_list.append(dist_shift)

            if "bg_aug_target" in data_dict:
                fg_image = data_dict["fg_image"].to(
                    self.device, non_blocking=True
                )
                bg_image = data_dict["bg_image"].to(
                    self.device, non_blocking=True
                )
                mask = data_dict["mask"].to(self.device, non_blocking=True)
                bg_aug_target = data_dict["bg_aug_target"]
                mixed_image = fg_image * mask + bg_image * (1 - mask)
                concat_image_list.append(mixed_image)
                concat_target_list.append(bg_aug_target)

                bg_aug_domain = data_dict["bg_aug_dist_shift"]
                concat_dist_shift_list.append(bg_aug_domain)

            image = torch.cat(concat_image_list, dim=0)
            target = torch.cat(concat_target_list, dim=0).to(
                self.device, non_blocking=True
            )
            dist_shift = torch.cat(concat_dist_shift_list, dim=0).to(
                self.device, non_blocking=True
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(image)
                output_per_domain, domain_output = self.classifier(feature)
                output = output_per_domain[
                    range(output_per_domain.shape[0]), dist_shift
                ]
                loss_cls = self.criterion(output, target)
                loss_domain = self.criterion(domain_output, dist_shift)
                loss = loss_cls + loss_domain

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss_cls.item(), image.size(0))
            losses_domain.update(loss_domain.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f},"
                f" loss_d: {losses_domain.avg:.4f}"
            )

        self.log_to_wandb(
            {"loss": losses.avg, "loss_domain": losses_domain.avg}
        )
