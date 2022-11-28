"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from JTT:
# https://github.com/anniesch/jtt
# --------------------------------------------------------

import torch


from .erm import ERMTrainer
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm
from utils import AverageMeter


class JTTTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "jtt"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_T_{args.jtt_id_epoch}"
            f"_up_{args.jtt_up_weight}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        args = self.args
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        return train_loader

    def infer_error_set(self):
        # train ERM model with T epoch
        args = self.args

        self.classifier.train()
        for idx_jtt_epoch in range(1, args.jtt_id_epoch + 1):
            losses = AverageMeter("Loss", ":.4e")
            pbar = tqdm(self.train_loader, dynamic_ncols=True)
            for data_dict in pbar:
                image, target = data_dict["image"], data_dict["target"]
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

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
                    f"ERM_ref [{idx_jtt_epoch}/{args.jtt_id_epoch}] loss: {losses.avg:.4f}"
                )

        error_set_list = []

        ordered_train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            shuffle=False,  # no shuffle for inferring error set
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

        pbar = tqdm(
            ordered_train_loader, dynamic_ncols=True, desc="inference for prediction"
        )
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["target"]
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    feature = self.backbone(image)
                    logits = self.classifier(feature)
                pred = logits.argmax(dim=1)
                error = pred != target
                error_set_list.append(error.long().cpu())

        error_set_list = torch.cat(error_set_list, dim=0)

        error_indices = torch.nonzero(error_set_list).flatten().tolist()
        self.error_indices = error_indices

        train_set = self.train_set
        upsampled_points = Subset(train_set,
                                  error_indices * args.jtt_up_weight)
        concat_train_set = ConcatDataset([train_set, upsampled_points])
        train_loader = torch.utils.data.DataLoader(
            concat_train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = train_loader

        # reinitialize the classifier and optimizer
        self._setup_models()
        self._setup_optimizers()

    def _before_train(self):
        self.infer_error_set()

    def _state_dict_for_save(self):
        state_dict = super()._state_dict_for_save()
        state_dict["error_indices"] = self.error_indices
        return state_dict

    def _load_state_dict(self, state_dict):
        super()._load_state_dict(state_dict)
        self.error_indices = state_dict["error_indices"]
