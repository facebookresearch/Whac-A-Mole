"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.transforms as transforms


from .base_trainer import BaseTrainer
from model.classifiers import LastLayerEnsemble, get_classifier
from utils import AverageMeter
from tqdm import tqdm
from dataset.urbancars_aug import UrbanCarsAug, urbancars_aug_collate
from model.transforms import CoRandomHorizontalFlip


class LLETrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "lle"

        default_name = (
            f"{args.method}"
            f"_es_{args.early_stop_metric}_{args.dataset}"
        )
        self.default_name = default_name

    def _setup_models(self):
        args = self.args
        backbone = get_classifier(args.arch, self.num_class)
        self.classifier = LastLayerEnsemble(
            self.num_class,
            num_dist_shift=3,
            backbone=backbone,
        ).to(self.device)

    def _modify_train_set(self, train_dataset):
        self.len_original_train_set = len(train_dataset)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        bg_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        co_transform = CoRandomHorizontalFlip()
        final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        mask_final_transform = transforms.ToTensor()

        bg_aug_train_set = UrbanCarsAug(
            "data",
            "bg",
            co_transform=co_transform,
            bg_transform=bg_transform,
            final_transform=final_transform,
            mask_final_transform=mask_final_transform,
        )

        co_occur_obj_aug_train_set = UrbanCarsAug(
            "data",
            "co_occur_obj",
            co_transform=co_transform,
            bg_transform=bg_transform,
            final_transform=final_transform,
            mask_final_transform=mask_final_transform,
        )
        concat_dataset_list = [train_dataset, bg_aug_train_set, co_occur_obj_aug_train_set]

        concat_dataset = torch.utils.data.ConcatDataset(concat_dataset_list)
        return concat_dataset

    def _get_train_loader(self, train_set):
        args = self.args

        weights = [1] * len(train_set)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, self.len_original_train_set, replacement=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            sampler=sampler,
            persistent_workers=args.num_workers > 0,
            collate_fn=urbancars_aug_collate,
        )
        return train_loader

    def _set_train(self):
        self.classifier.train()

    def train(self):
        args = self.args
        self._set_train()
        losses = AverageMeter("Loss", ":.4e")
        losses_dist_shift = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:

            concat_image_list = []
            concat_target_list = []
            concat_dist_shift_list = []

            if "image" in data_dict:
                image, target = data_dict["image"], data_dict["label"]
                dist_shift_label = data_dict["dist_shift"]

                obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
                image = image.to(self.device, non_blocking=True)
                concat_image_list.append(image)
                concat_target_list.append(obj_gt)
                concat_dist_shift_list.append(dist_shift_label)

            if "aug_image" in data_dict:
                obj_image = data_dict["aug_image"].to(
                    self.device, non_blocking=True
                )
                bg_image = data_dict["aug_bg_image"].to(
                    self.device, non_blocking=True
                )
                co_occur_obj_image = data_dict["aug_co_occur_obj_image"].to(
                    self.device, non_blocking=True
                )
                obj_mask = data_dict["aug_obj_mask"].to(
                    self.device, non_blocking=True
                )
                co_occur_obj_mask = data_dict["aug_co_occur_obj_mask"].to(
                    self.device, non_blocking=True
                )

                img_w_co_obj = (
                    co_occur_obj_image * co_occur_obj_mask
                    + bg_image * (1 - co_occur_obj_mask)
                )
                mixed_img = obj_image * obj_mask + img_w_co_obj * (1 - obj_mask)

                aug_label = data_dict["aug_label"]
                aug_obj_gt = aug_label[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
                aug_dist_shift_label = data_dict["aug_dist_shift"]

                concat_image_list.append(mixed_img)
                concat_target_list.append(aug_obj_gt)
                concat_dist_shift_list.append(aug_dist_shift_label)

            image = torch.cat(concat_image_list, dim=0)
            obj_gt = torch.cat(concat_target_list, dim=0).to(
                self.device, non_blocking=True
            )
            dist_shift_label = torch.cat(concat_dist_shift_list, dim=0).to(
                self.device, non_blocking=True
            )

            with torch.cuda.amp.autocast(enabled=args.amp):
                output_per_dist_shift, dist_shift_output = self.classifier(image)
                output = output_per_dist_shift[
                    range(output_per_dist_shift.shape[0]), dist_shift_label
                ]
                loss_cls = self.criterion(output, obj_gt)
                loss_dist_shift = self.criterion(dist_shift_output, dist_shift_label)
                loss = loss_cls + loss_dist_shift

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss_cls.item(), image.size(0))
            losses_dist_shift.update(loss_dist_shift.item(), image.size(0))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f},"
                f" loss_d: {losses_dist_shift.avg:.4f}"
            )

        self.log_to_wandb(
            {"loss": losses.avg, "loss_dist_shift": losses_dist_shift.avg}
        )
