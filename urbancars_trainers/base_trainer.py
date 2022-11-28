"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import submitit
import wandb
import torch
import copy
import torch.nn as nn


from tqdm import tqdm
from dataset.urbancars import UrbanCars
from model.classifiers import (
    get_classifier,
    get_transforms,
)
from utils import (
    set_seed,
    MultiDimAverageMeter,
)


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self._setup_method_name_and_default_name()

        self.cur_epoch = 1

        if args.run_name is None:
            args.run_name = self.default_name
        else:
            args.run_name += f"_{self.default_name}"
        ckpt_dir = os.path.join(
            args.exp_root, args.run_name, f"seed_{args.seed}"
        )

        if args.resume is None:
            os.makedirs(ckpt_dir, exist_ok=True)

        print("ckpt_dir: ", ckpt_dir)
        self.ckpt_dir = ckpt_dir

        self.ckpt_fname = "ckpt"

        self.cond_best_acc = 0
        self.cond_on_best_val_log_dict = {}

    def _setup_all(self):
        args = self.args
        set_seed(args.seed)
        self.device = torch.device(0)

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self._setup_early_stop_metric()
        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._method_specific_setups()

        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.run_name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
            )

        # loading checkpoint
        if args.resume:
            ckpt_fpath = args.resume
            assert os.path.exists(ckpt_fpath), f"{ckpt_fpath} does not exist"
            state_dict = torch.load(ckpt_fpath, map_location="cpu")
            self._load_state_dict(state_dict)
        else:
            self._before_train()

    def _get_train_collate_fn(self):
        return None

    def _get_train_loader(self, train_set):
        args = self.args
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            collate_fn=self._get_train_collate_fn(),
        )
        return train_loader

    def _setup_early_stop_metric(self):
        args = self.args

        early_stop_metric_arg_to_real_metric = {
            "bg": "val_bg_worst_group_acc",
            "co_occur_obj": "val_co_occur_obj_worst_group_acc",
            "both": "val_both_worst_group_acc",
        }

        if args.method in [
            "groupdro",
            "di",
            "subg",
            "dfr",
        ]:
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.group_label
            ]
        elif args.method in [
            "erm",
            "lff",
            "eiil",
            "sd",
            "jtt",
            "debian",
            "lle",
            "cf_f_aug",
            "augmix",
            "cutmix",
            "mixup",
            "cutout",
        ]:
            # methods that do not use group labels
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.early_stop_metric
            ]
        else:
            raise ValueError(f"unknown method: {args.method}")

    def _get_train_transform(self):
        args = self.args
        train_transform = get_transforms(args.arch, is_training=True)
        return train_transform

    def _setup_dataset(self):
        args = self.args

        train_transform = self._get_train_transform()
        test_transform = get_transforms(args.arch, is_training=False)

        train_set = UrbanCars(
            "data",
            "train",
            group_label=args.group_label,
            transform=train_transform,
            return_group_index=args.method in ["groupdro", "eiil"],
            return_domain_label=args.method == "di",
            return_dist_shift=args.method == "lle",
        )
        self.train_set = train_set
        val_set = UrbanCars(
            "data",
            "val",
            transform=test_transform,
        )
        test_set = UrbanCars(
            "data",
            "test",
            transform=test_transform,
        )
        self.obj_name_list = train_set.obj_name_list
        self.num_class = len(self.obj_name_list)

        train_set = self._modify_train_set(train_set)
        train_loader = self._get_train_loader(train_set)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _method_specific_setups(self):
        pass

    def _setup_models(self):
        args = self.args
        self.classifier = get_classifier(
            args.arch,
            self.num_class,
        ).to(self.device)

    def _set_train(self):
        self.classifier.train()

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_optimizers(self):
        args = self.args
        parameters = [
            p for p in self.classifier.parameters() if p.requires_grad
        ]
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _before_train(self):
        pass

    def __call__(self):
        torch.backends.cudnn.benchmark = True
        args = self.args
        self._setup_all()

        for _ in range(self.cur_epoch, args.epoch + 1):
            self.train()
            is_best = self.eval()
            state_dict = self._state_dict_for_save()
            self._save_ckpt(state_dict, self.ckpt_fname)
            if is_best:
                self._save_ckpt(state_dict, "best")

            id_acc = self.cond_on_best_val_log_dict["cond_test_id_acc"] * 100
            bg_gap = self.cond_on_best_val_log_dict["cond_test_bg_gap"] * 100
            co_occur_obj_gap = self.cond_on_best_val_log_dict[
                "cond_test_co_occur_obj_gap"
            ] * 100
            both_gap = self.cond_on_best_val_log_dict["cond_test_both_gap"] * 100
            print(
                f"[{self.cur_epoch}/{args.epoch}] "
                f"ID Acc: {id_acc:.2f} "
                f"BG gap: {bg_gap:.2f} "
                f"CoObj gap: {co_occur_obj_gap:.2f} "
                f"BG+CoObj gap: {both_gap:.2f}"
            )
            self.cur_epoch += 1

    def train(self):
        raise NotImplementedError

    def eval(self):
        val_log_dict = self._eval_split(self.val_loader, "val")
        test_log_dict = self._eval_split(self.test_loader, "test")

        early_stop_metric_result = val_log_dict[
            self.args.early_stop_metric_real
        ]

        if (
            early_stop_metric_result <= self.cond_best_acc
            and self.cur_epoch > 1
            and len(self.cond_on_best_val_log_dict) > 0
        ):
            self.log_to_wandb(self.cond_on_best_val_log_dict)
            return False  # not best

        is_best = False

        if early_stop_metric_result > self.cond_best_acc:
            self.cond_best_acc = early_stop_metric_result
            is_best = True
            for key, value in test_log_dict.items():
                new_key = f"cond_{key}"
                self.cond_on_best_val_log_dict[new_key] = value

        self.log_to_wandb(self.cond_on_best_val_log_dict)
        return is_best

    @torch.no_grad()
    def _eval_split(self, loader, split):
        args = self.args

        meter = MultiDimAverageMeter(
            (self.num_class, self.num_class, self.num_class)
        )
        total_correct = []
        total_bg_correct = []
        total_co_occur_obj_correct = []
        total_shortcut_conflict_mask = []

        self.classifier.eval()
        pbar = tqdm(loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)

            pred = output.argmax(dim=1)

            obj_label = target[:, 0]
            bg_label = target[:, 1]
            co_occur_obj_label = target[:, 2]

            shortcut_conflict_mask = bg_label != co_occur_obj_label
            total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

            correct = pred == obj_label
            meter.add(correct.cpu(), target.cpu())
            total_correct.append(correct.cpu())

            bg_correct = pred == bg_label
            total_bg_correct.append(bg_correct.cpu())

            co_occur_obj_correct = pred == co_occur_obj_label
            total_co_occur_obj_correct.append(co_occur_obj_correct.cpu())

        num_correct = meter.cum.reshape(*meter.dims)
        cnt = meter.cnt.reshape(*meter.dims)
        multi_dim_color_acc = num_correct / cnt
        log_dict = {}
        absent_present_str_list = ["absent", "present"]
        absent_present_bg_ratio_list = [1 - args.bg_ratio, args.bg_ratio]
        absent_present_co_occur_obj_ratio_list = [
            1 - args.co_occur_obj_ratio,
            args.co_occur_obj_ratio,
        ]

        weighted_group_acc = 0
        for bg_shortcut in range(len(absent_present_str_list)):
            for second_shortcut in range(len(absent_present_str_list)):
                first_shortcut_mask = (meter.eye_tsr == bg_shortcut).unsqueeze(2)
                co_occur_obj_shortcut_mask = (
                    meter.eye_tsr == second_shortcut
                ).unsqueeze(1)
                mask = first_shortcut_mask * co_occur_obj_shortcut_mask
                acc = multi_dim_color_acc[mask].mean().item()
                bg_shortcut_str = absent_present_str_list[bg_shortcut]
                co_occur_obj_shortcut_str = absent_present_str_list[
                    second_shortcut
                ]
                log_dict[
                    f"{split}_bg_{bg_shortcut_str}"
                    f"_co_occur_obj_{co_occur_obj_shortcut_str}_acc"
                ] = acc
                cur_group_bg_ratio = absent_present_bg_ratio_list[bg_shortcut]
                cur_group_co_occur_obj_ratio = (
                    absent_present_co_occur_obj_ratio_list[second_shortcut]
                )
                cur_group_ratio = (
                    cur_group_bg_ratio * cur_group_co_occur_obj_ratio
                )
                weighted_group_acc += acc * cur_group_ratio

        bg_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_present_acc"]
            - weighted_group_acc
        )
        co_occur_obj_gap = (
            log_dict[f"{split}_bg_present_co_occur_obj_absent_acc"]
            - weighted_group_acc
        )
        both_gap = (
            log_dict[f"{split}_bg_absent_co_occur_obj_absent_acc"]
            - weighted_group_acc
        )

        log_dict.update(
            {
                f"{split}_id_acc": weighted_group_acc,
                f"{split}_bg_gap": bg_gap,
                f"{split}_co_occur_obj_gap": co_occur_obj_gap,
                f"{split}_both_gap": both_gap,
            }
        )

        total_bg_correct = torch.cat(total_bg_correct, dim=0)
        total_co_occur_obj_correct = torch.cat(
            total_co_occur_obj_correct, dim=0
        )
        total_correct = torch.cat(total_correct, dim=0)

        (
            bg_worst_group_acc,
            co_occur_obj_worst_group_acc,
            both_worst_group_acc,
        ) = meter.get_worst_group_acc()

        log_dict.update(
            {
                f"{split}_bg_worst_group_acc": bg_worst_group_acc,
                f"{split}_co_occur_obj_worst_group_acc": co_occur_obj_worst_group_acc,
                f"{split}_both_worst_group_acc": both_worst_group_acc,
            }
        )

        if args.method == "erm":
            # evaluate cue preference for ERM
            obj_acc = total_correct.float().mean().item()
            bg_acc = total_bg_correct.float().mean().item()
            co_occur_obj_acc = total_co_occur_obj_correct.float().mean().item()

            log_dict.update(
                {
                    f"{split}_cue_obj_acc": obj_acc,
                    f"{split}_cue_bg_acc": bg_acc,
                    f"{split}_cue_co_occur_obj_acc": co_occur_obj_acc,
                }
            )

        self.log_to_wandb(log_dict)

        return log_dict

    def _state_dict_for_save(self):
        state_dict = {
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
            "cond_best_acc": self.cond_best_acc,
            "cond_on_best_val_log_dict": self.cond_on_best_val_log_dict,
        }
        return state_dict

    def _load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict["scaler"])
        self.cur_epoch = state_dict["epoch"] + 1
        self.classifier.load_state_dict(state_dict["classifier"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.cond_best_acc = state_dict["cond_best_acc"]
        self.cond_on_best_val_log_dict = state_dict["cond_on_best_val_log_dict"]

    def _save_ckpt(self, state_dict, name):
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save(state_dict, ckpt_fpath)

    def _loss_backward(self, loss, retain_graph=False):
        if self.args.amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def checkpoint(self):
        new_args = copy.deepcopy(self.args)
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{self.ckpt_fname}.pth")
        if os.path.exists(ckpt_fpath):
            new_args.resume = ckpt_fpath
        training_callable = self.__class__(new_args)
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable)

    def log_to_wandb(self, log_dict, step=None):
        if step is None:
            step = self.cur_epoch
        if self.args.wandb:
            wandb.log(log_dict, step=step)
