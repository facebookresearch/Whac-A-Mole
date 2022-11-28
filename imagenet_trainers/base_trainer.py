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
import torchvision.transforms as transforms
import copy
import torch.nn as nn
import torchvision.models as torchvision_models
import math
import model.mae_vits as mae_vits
import torchvision.models as tv_models


from tqdm import tqdm
from utils import set_seed
from dataset.imagenet import ImageNet
from dataset.imagenet_a import ImageNetA
from dataset.objectnet import ObjectNet
from dataset.imagenet_r import ImageNetR
from dataset.imagenet_200 import ImageNet200
from dataset.imagenetv2 import ImageNetV2
from dataset.imagenet9 import ImageNet9
from dataset.imagenet_sketch import ImageNetSketch
from dataset.imagenet_d import ImageNetD
from dataset.imagenet_stylized import ImageNetStylized
from imagenet_w.watermark_transform import AddWatermark
from model.classifiers import LastLayerEnsemble
from model.optimizer import LARS
from imagenet_w.watermark_transform import CARTON_CLASS_INDEX


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self._setup_method_name_and_default_name()
        self.cur_epoch = 1

        if args.run_name is None:
            args.run_name = self.default_name
        else:
            args.run_name += f"_{self.default_name}"
        self.args.run_name = args.run_name

        ckpt_dir = os.path.join(
            args.exp_root, args.run_name, f"seed_{args.seed}"
        )

        if not args.evaluate:
            os.makedirs(ckpt_dir, exist_ok=True)
        print("ckpt_dir: ", ckpt_dir)
        self.ckpt_dir = ckpt_dir
        self.ckpt_fname = "ckpt"

    def _setup_all(self):
        torch.backends.cudnn.benchmark = True

        args = self.args
        set_seed(args.seed)
        self.device = torch.device(0)

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self._setup_models()
        self._setup_dataset()
        self._setup_criterion()
        self._setup_optimizers()
        self._method_specific_setups()

        # loading checkpoint
        if args.resume:
            # ckpt_fpath = os.path.join(self.ckpt_dir, f"{self.ckpt_fname}.pth")
            ckpt_fpath = args.resume
            assert os.path.exists(ckpt_fpath), f"{ckpt_fpath} does not exist"
            state_dict = torch.load(ckpt_fpath, map_location="cpu")
            self._load_state_dict(state_dict)
            print(f"resume from: {ckpt_fpath}")
        else:
            self._before_train()

        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.run_name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
            )

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

    def _get_test_loader(self, val_set):
        args = self.args
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        return val_loader

    def _get_normalize_mean_std(self):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def _get_resize_and_crop_size(self):
        args = self.args

        (arch, pretrained_approach) = args.feature_extractor.split("_")
        if pretrained_approach == "swag-ft":
            assert arch == "vit-b"
            resize_size = 384
            crop_size = 384
        else:
            resize_size = 256
            crop_size = 224

        return resize_size, crop_size

    def _get_train_transform(self):
        args = self.args

        mean, std = self._get_normalize_mean_std()
        normalize = transforms.Normalize(mean=mean, std=std)
        crop_size = self._get_resize_and_crop_size()[1]

        (arch, pretrained_approach) = args.feature_extractor.split("_")
        if pretrained_approach == "swag-ft":
            assert arch == "vit-b"
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        crop_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            assert pretrained_approach in ["erm", "mae-ft"]

            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return train_transform

    def _get_test_transform(self):
        args = self.args
        mean, std = self._get_normalize_mean_std()
        normalize = transforms.Normalize(mean=mean, std=std)
        resize_size, crop_size = self._get_resize_and_crop_size()
        (arch, pretrained_approach) = args.feature_extractor.split("_")

        if pretrained_approach == "swag-ft":
            assert arch == "vit-b"
            if arch == "vit-b":
                test_transform = (
                    tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
                )
            elif arch == "vit-l":
                test_transform = (
                    tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
                )
            elif arch == "vit-h":
                test_transform = (
                    tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
                )
            else:
                raise NotImplementedError
        else:
            assert pretrained_approach in ["erm", "mae-ft"]
            test_transform = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return test_transform

    def _setup_dataset(self):
        args = self.args

        train_transform = self._get_train_transform()
        test_transform = self._get_test_transform()

        train_set = ImageNet(
            args.data_root,
            "train",
            transform=train_transform,
            return_group_index=args.method == "eiil"
            and args.resume is not None,
            return_dist_shift_index=args.method == "lle",
        )
        self.train_set = train_set
        val_set = ImageNet(
            args.data_root,
            "val",
            transform=test_transform,
        )
        self.class_name_list = train_set.classes
        self.num_class = len(self.class_name_list)

        train_set = self._modify_train_set(train_set)
        train_loader = self._get_train_loader(train_set)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _method_specific_setups(self):
        pass

    def _setup_models(self):
        args = self.args

        (arch, pretrained_approach) = args.feature_extractor.split("_")
        if pretrained_approach == "erm":
            assert arch == "resnet50"
            model = torchvision_models.__dict__[arch](weights="IMAGENET1K_V1")
            linear_keyword = "fc"
        elif pretrained_approach == "mae-ft":
            arch_to_func_name = {
                "vit-b": "vit_base_patch16",
                "vit-l": "vit_large_patch16",
                "vit-h": "vit_huge_patch14",
            }
            func_name = arch_to_func_name[arch]
            model = mae_vits.__dict__[func_name](
                num_classes=1000,
                global_pool=True,
            )
            linear_keyword = "head"
        elif pretrained_approach == "swag-ft":
            assert arch == "vit-b"
            if arch == "vit-b":
                model = tv_models.vit_b_16(
                    weights=tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                )
            elif arch == "vit-l":
                model = tv_models.vit_l_16(
                    weights=tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
                )
            elif arch == "vit-h":
                model = tv_models.vit_h_14(
                    weights=tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
                )
            else:
                raise NotImplementedError

            linear_keyword = "heads"
        else:
            raise NotImplementedError

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in [
                "%s.weight" % linear_keyword,
                "%s.bias" % linear_keyword,
            ]:
                param.requires_grad = False

        if pretrained_approach == "mae-ft":
            mae_ckpt_fpath = f"exp/weights/mae_{arch}_ft.pth"
            checkpoint = torch.load(mae_ckpt_fpath, map_location="cpu")
            ckpt_state_dict = checkpoint["model"]

            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in ckpt_state_dict
                    and ckpt_state_dict[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del ckpt_state_dict[k]

            # interpolate position embedding
            mae_vits.interpolate_pos_embed(model, ckpt_state_dict)

            # load pre-trained model
            model.load_state_dict(ckpt_state_dict, strict=True)

        if pretrained_approach == "swag-ft":
            linear_weight_shape = getattr(model, linear_keyword)[0].weight.shape
            linear_state_dict = getattr(model, linear_keyword)[0].state_dict()
        else:
            linear_weight_shape = getattr(model, linear_keyword).weight.shape
            linear_state_dict = getattr(model, linear_keyword).state_dict()

        # it will be used for other models, e.g., bias amplified network in LfF
        self.linear_weight_shape = linear_weight_shape

        setattr(model, linear_keyword, nn.Identity())
        self.backbone = model.to(self.device)
        self.backbone.eval()  # not changing batch norm weights during linear probing

        if args.method == "lle":
            self.classifier = LastLayerEnsemble(
                num_classes=linear_weight_shape[0],
                num_dist_shift=args.num_dist_shift,
                in_features=linear_weight_shape[1],
            ).to(self.device)
            for linear_layer in self.classifier.ensemble_classifier_list:
                linear_layer.load_state_dict(linear_state_dict)
        else:
            self.classifier = nn.Linear(
                linear_weight_shape[1], linear_weight_shape[0]
            ).to(self.device)
            self.classifier.load_state_dict(linear_state_dict)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_optimizers(self):
        args = self.args
        self.init_lr = args.lr * args.batch_size / 256

        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.classifier.parameters(),
                self.init_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "lars":
            self.optimizer = LARS(
                self.classifier.parameters(),
                lr=self.init_lr,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError

    def adjust_learning_rate(self, optimizer, init_lr, epoch):
        args = self.args

        if args.feature_extractor.endswith("_mae-ft"):
            # no need to adjust learning rate when using mae
            return

        """Decay the learning rate based on schedule"""
        args = self.args
        cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epoch))
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _before_train(self):
        pass

    def __call__(self):
        args = self.args
        self._setup_all()

        if args.evaluate:
            self.eval()
            return

        for e in range(self.cur_epoch, args.epoch + 1):
            self.adjust_learning_rate(self.optimizer, self.init_lr, e - 1)
            self.train()
            self.eval()
            state_dict = self._state_dict_for_save()
            self._save_ckpt(state_dict, self.ckpt_fname)
            self.cur_epoch += 1

    def train(self):
        raise NotImplementedError

    def eval(self):
        args = self.args
        in_1k_acc, in_1k_carton_acc = self._eval_split(
            self.val_loader, "imagenet_1k"
        )
        log_dict = {"in_1k_acc": in_1k_acc}

        if self.cur_epoch < args.epoch and not args.evaluate:
            return log_dict

        # evaluate on IN-W
        in_w_dataset = self._get_ood_dataset("imagenet-w")
        map_prediction_func = getattr(in_w_dataset, "map_prediction", None)
        in_w_loader = self._get_test_loader(in_w_dataset)
        in_w_acc, in_w_carton_acc = self._eval_ood(
            in_w_loader, "imagenet-w", map_prediction_func
        )
        in_w_gap = in_w_acc - in_1k_acc
        carton_gap = in_w_carton_acc - in_1k_carton_acc
        log_dict["in_w_gap"] = in_w_gap
        log_dict["carton_gap"] = carton_gap

        # evaluate on SIN
        sin_dataset = self._get_ood_dataset("imagenet-stylized")
        map_prediction_func = getattr(sin_dataset, "map_prediction", None)
        sin_loader = self._get_test_loader(sin_dataset)
        sin_acc = self._eval_ood(
            sin_loader, "imagenet-stylized", map_prediction_func
        )
        sin_gap = sin_acc - in_1k_acc
        log_dict["sin_gap"] = sin_gap

        # evaluate on IN-R
        in_r_dataset = self._get_ood_dataset("imagenet-r")
        map_prediction_func = getattr(in_r_dataset, "map_prediction", None)
        in_r_loader = self._get_test_loader(in_r_dataset)
        in_r_acc = self._eval_ood(
            in_r_loader, "imagenet-r", map_prediction_func
        )

        in_200_dataset = self._get_ood_dataset("imagenet-200")
        map_prediction_func = getattr(in_200_dataset, "map_prediction", None)
        in_200_loader = self._get_test_loader(in_200_dataset)
        in_200_acc = self._eval_ood(
            in_200_loader, "imagenet-200", map_prediction_func
        )

        in_r_gap = in_r_acc - in_200_acc
        log_dict["in_r_gap"] = in_r_gap

        # evaluate on IN-9 (background challenge)
        in_9_mixed_same_dataset = self._get_ood_dataset("mixed_same")
        map_prediction_func = getattr(
            in_9_mixed_same_dataset, "map_prediction", None
        )
        in_9_mixed_same_loader = self._get_test_loader(in_9_mixed_same_dataset)
        in_9_mixed_same_acc = self._eval_ood(
            in_9_mixed_same_loader, "imagenet-9_mixed_same", map_prediction_func
        )

        in_9_mixed_rand_dataset = self._get_ood_dataset("mixed_rand")
        map_prediction_func = getattr(
            in_9_mixed_rand_dataset, "map_prediction", None
        )
        in_9_mixed_rand_loader = self._get_test_loader(in_9_mixed_rand_dataset)
        in_9_mixed_rand_acc = self._eval_ood(
            in_9_mixed_rand_loader, "imagenet-9_mixed_rand", map_prediction_func
        )

        in_9_gap = in_9_mixed_rand_acc - in_9_mixed_same_acc
        log_dict["in_9_gap"] = in_9_gap

        if args.eval_in_sketch:
            # evaluate IN-sketch
            in_sketch_dataset = self._get_ood_dataset("imagenet-sketch")
            map_prediction_func = getattr(
                in_sketch_dataset, "map_prediction", None
            )
            in_sketch_loader = self._get_test_loader(in_sketch_dataset)
            in_sketch_acc = self._eval_ood(
                in_sketch_loader, "imagenet-sketch", map_prediction_func
            )
            in_sketch_gap = in_sketch_acc - in_1k_acc
            log_dict["in_sketch_gap"] = in_sketch_gap

        if args.eval_other_ood:
            # evaluate on IN-A, INV2, ObjectNet, and IN-D
            for ood_dataset_name in [
                "imagenet-a",
                "imagenetv2",
                "objectnet",
                "imagenet-d_clipart",
                "imagenet-d_infograph",
                "imagenet-d_painting",
                "imagenet-d_quickdraw",
                "imagenet-d_real",
                "imagenet-d_sketch",
            ]:
                ood_dataset = self._get_ood_dataset(ood_dataset_name)
                map_prediction_func = getattr(
                    ood_dataset, "map_prediction", None
                )
                ood_loader = self._get_test_loader(ood_dataset)
                ood_acc = self._eval_ood(
                    ood_loader, ood_dataset_name, map_prediction_func
                )
                log_dict[f"{ood_dataset_name}_acc"] = ood_acc

            # compute mDE (https://arxiv.org/abs/2104.12928)
            alexnet_in_d_error = {
                "imagenet-d_clipart": 0.84010,
                "imagenet-d_infograph": 0.95072,
                "imagenet-d_painting": 79.080,
                "imagenet-d_quickdraw": 99.745,
                "imagenet-d_real": 0.54887,
                "imagenet-d_sketch": 0.91189,
            }
            in_d_sum_domain_error = 0
            for in_d_dataset_name in [
                "imagenet-d_clipart",
                "imagenet-d_infograph",
                "imagenet-d_painting",
                "imagenet-d_quickdraw",
                "imagenet-d_real",
                "imagenet-d_sketch"]:
                in_d_domain_acc = log_dict[f"{in_d_dataset_name}_acc"]
                in_d_domain_error = 1 - in_d_domain_acc
                nomalized_in_d_domain_error = in_d_domain_error / alexnet_in_d_error[in_d_dataset_name]
                in_d_sum_domain_error += nomalized_in_d_domain_error
            mde = in_d_sum_domain_error / len([
                "imagenet-d_clipart",
                "imagenet-d_infograph",
                "imagenet-d_painting",
                "imagenet-d_quickdraw",
                "imagenet-d_real",
                "imagenet-d_sketch"])
            log_dict["imagenet_d_mde"] = mde

        for k, v in log_dict.items():
            print(f"{k}: {v:.4f}")

        return log_dict

    def _get_ood_dataset(self, ood_dataset_name):
        args = self.args

        resize_size, crop_size = self._get_resize_and_crop_size()
        test_transform = self._get_test_transform()

        if ood_dataset_name == "imagenet-w":
            mean, std = self._get_normalize_mean_std()
            watermark_transform = transforms.Compose(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    AddWatermark(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            dataset = ImageNet(
                args.data_root,
                "val",
                transform=watermark_transform,
            )
        elif ood_dataset_name == "imagenet-stylized":
            dataset = ImageNetStylized(
                args.data_root, "val", transform=test_transform
            )
        elif ood_dataset_name == "imagenet-200":
            dataset = ImageNet200(args.data_root, transform=test_transform)
        elif ood_dataset_name == "imagenet-r":
            dataset = ImageNetR(args.data_root, transform=test_transform)
        elif ood_dataset_name in ["mixed_same", "mixed_rand"]:
            dataset = ImageNet9(
                args.data_root,
                ood_dataset_name,
                transform=test_transform,
            )
        elif ood_dataset_name == "imagenet-sketch":
            dataset = ImageNetSketch(args.data_root, transform=test_transform)
        elif ood_dataset_name == "imagenet-a":
            dataset = ImageNetA(args.data_root, transform=test_transform)
        elif ood_dataset_name.startswith("imagenetv2"):
            dataset = ImageNetV2(
                args.data_root,
                transform=test_transform,
            )
        elif ood_dataset_name == "objectnet":
            dataset = ObjectNet(args.data_root, transform=test_transform)
        elif ood_dataset_name.startswith("imagenet-d"):
            dataset = ImageNetD(
                args.data_root,
                ood_dataset_name,
                transform=test_transform,
            )
        else:
            raise NotImplementedError

        return dataset

    @torch.no_grad()
    def _eval_ood(self, loader, ood_dataset_name, map_prediction_func):
        args = self.args

        total_pred = []
        total_label = []

        self.classifier.eval()
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"eval {ood_dataset_name}")
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["target"]
            image = image.to(self.device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                feature = self.backbone(image)
                output = self.classifier(feature)

            if ood_dataset_name in [
                "imagenet-a",
                "imagenet-r",
                "imagenet-200",
                "objectnet",
            ]:
                output = map_prediction_func(output)
                pred = output.argmax(dim=1)
            elif ood_dataset_name.startswith(
                "imagenetv2"
            ) or ood_dataset_name in [
                "imagenet-sketch",
                "imagenet-w",
                "imagenet-stylized",
            ]:
                pred = output.argmax(dim=1)
            elif ood_dataset_name.startswith(
                "imagenet-9"
            ) or ood_dataset_name.startswith("imagenet-d"):
                pred = output.argmax(dim=1)
                pred = map_prediction_func(pred)
            else:
                raise NotImplementedError
            total_pred.append(pred.cpu())
            total_label.append(target)

        total_pred = torch.cat(total_pred, dim=0).numpy()
        total_label = torch.cat(total_label, dim=0).numpy()
        acc = ((total_pred == total_label).sum() / len(total_label)).item()

        if ood_dataset_name == "imagenet-w":
            carton_class_mask = total_label == CARTON_CLASS_INDEX
            carton_pred = total_pred[carton_class_mask]
            carton_acc = (carton_pred == CARTON_CLASS_INDEX).mean()
            return acc, carton_acc
        else:
            return acc

    @torch.no_grad()
    def _eval_split(self, loader, split):
        args = self.args

        total_pred = []
        total_label = []

        self.classifier.eval()
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"eval {split}")
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["target"]
            image = image.to(self.device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                feature = self.backbone(image)
                output = self.classifier(feature)

            pred = output.argmax(dim=1)
            total_pred.append(pred.cpu())
            total_label.append(target)

        total_pred = torch.cat(total_pred, dim=0).numpy()
        total_label = torch.cat(total_label, dim=0).numpy()
        acc = ((total_pred == total_label).sum() / len(total_label)).item()

        carton_class_mask = total_label == CARTON_CLASS_INDEX
        carton_pred = total_pred[carton_class_mask]
        carton_acc = (carton_pred == CARTON_CLASS_INDEX).mean()

        return acc, carton_acc

    def _state_dict_for_save(self):
        state_dict = {
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
        }
        return state_dict

    def _load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict["scaler"])
        self.cur_epoch = state_dict["epoch"] + 1
        self.classifier.load_state_dict(state_dict["classifier"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

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
        assert new_args.run_name is not None
        training_callable = self.__class__(new_args)
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable)

    def log_to_wandb(self, log_dict, step=None):
        if step is None:
            step = self.cur_epoch
        if self.args.wandb:
            wandb.log(log_dict, step=step)
