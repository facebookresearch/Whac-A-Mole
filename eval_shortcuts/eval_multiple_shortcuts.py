"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
import submitit
import wandb
import torch
import torchvision.transforms as transforms
import copy


from tqdm import tqdm
from utils import set_seed
from imagenet_w.watermark_transform import AddWatermark, CARTON_CLASS_INDEX
from torchvision.transforms._presets import ImageClassification
from dataset.imagenet import ImageNet
from dataset.imagenet_stylized import ImageNetStylized
from dataset.imagenet_200 import ImageNet200
from dataset.imagenet_r import ImageNetR
from dataset.imagenet9 import ImageNet9
from dataset.imagenet_sketch import ImageNetSketch
from model.model_zoo import get_model_and_transforms
from utils import slurm_wandb_argparser


class MultiShortcutEvaluator:
    def __init__(self, args):
        self.args = args

    def _setup_all(self):
        torch.backends.cudnn.benchmark = True

        args = self.args
        set_seed(args.seed)
        self.device = torch.device(0)

        args.run_name = f"{args.model}_eval_multi_shortcut"
        self._setup_models(args.model)

        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.run_name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
            )

    def _setup_dataset(self, in_1k_transform, in_w_transform):
        args = self.args

        in_1k_val_set = ImageNet(
            args.data_root, "val", transform=in_1k_transform
        )
        in_w_val_set = ImageNet(args.data_root, "val", transform=in_w_transform)
        in_stylized = ImageNetStylized(
            args.data_root, "val", transform=in_1k_transform
        )
        in_200 = ImageNet200(args.data_root, transform=in_1k_transform)
        in_r = ImageNetR(args.data_root, transform=in_1k_transform)
        in_9_mixed_same = ImageNet9(
            args.data_root, "mixed_same", transform=in_1k_transform
        )
        in_9_mixed_rand = ImageNet9(
            args.data_root, "mixed_rand", transform=in_1k_transform
        )

        dataset_dict = {
            "in_1k": in_1k_val_set,
            "in_w": in_w_val_set,
            "in_stylized": in_stylized,
            "in_200": in_200,
            "in_r": in_r,
            "in_9_mixed_same": in_9_mixed_same,
            "in_9_mixed_rand": in_9_mixed_rand,
        }

        if args.eval_in_sketch:
            in_sketch = ImageNetSketch("data", transform=in_1k_transform)
            dataset_dict["in_sketch"] = in_sketch

        self.dataset_dict = dataset_dict
        self.num_class = 1000  # 1000 for ImageNet

    def _setup_models(self, model_name):
        print(f"Loading {model_name}...")
        model, in_1k_transform = get_model_and_transforms(
            model_name, self.device
        )

        # insert watermark transform into clean transforms
        if isinstance(in_1k_transform, transforms.Compose):
            crop_size = in_1k_transform.transforms[1].size[0]
            watermark_transform = AddWatermark(crop_size)
            in_w_transform_list = (
                in_1k_transform.transforms[:-1]
                + [watermark_transform]
                + [in_1k_transform.transforms[-1]]
            )
        elif isinstance(in_1k_transform, ImageClassification):
            crop_size = in_1k_transform.crop_size[0]
            watermark_transform = AddWatermark(crop_size)
            in_w_transform_list = [
                transforms.Resize(
                    in_1k_transform.resize_size,
                    interpolation=in_1k_transform.interpolation,
                ),
                transforms.CenterCrop(in_1k_transform.crop_size),
                transforms.ToTensor(),
                watermark_transform,
                transforms.Normalize(
                    mean=in_1k_transform.mean, std=in_1k_transform.std
                ),
            ]
        else:
            raise NotImplementedError

        in_w_transform = transforms.Compose(in_w_transform_list)
        self._setup_dataset(in_1k_transform, in_w_transform)
        self.model = model

    def __call__(self):
        self._setup_all()
        self.eval()

    def eval(self):
        args = self.args

        def dataset_to_loader(dataset):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                persistent_workers=args.num_workers > 0,
            )

        model_name = self.args.model

        # evaluate on IN-1k
        (in_1k_acc, in_1k_carton_acc,) = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_1k"]),
            model_name,
            "IN-1k",
        )

        # evaluate watermark shortcut reliance on IN-W
        (in_w_acc, in_w_carton_acc,) = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_w"]), model_name, "IN-W"
        )
        in_w_gap = in_w_acc - in_1k_acc
        carton_gap = in_w_carton_acc - in_1k_carton_acc

        # evaluate texture shortcut reliance on SIN (Stylized ImageNet)
        in_stylized_acc = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_stylized"]),
            model_name,
            "SIN",
        )
        sin_gap = in_stylized_acc - in_1k_acc

        # evaluate texture shortcut reliance on IN-R
        in_200_acc = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_200"]), model_name, "IN-200"
        )
        in_r_acc = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_r"]), model_name, "IN-R"
        )
        in_r_gap = in_r_acc - in_200_acc

        # evaluate background shortcut reliance on IN-9
        mixed_same_acc = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_9_mixed_same"]),
            model_name,
            "IN-9 (mixed same)",
        )
        mixed_rand_acc = self._eval_model(
            dataset_to_loader(self.dataset_dict["in_9_mixed_rand"]),
            model_name,
            "IN-9 (mixed rand)",
        )
        in_9_gap = mixed_rand_acc - mixed_same_acc

        log_dict = {
            "in_1k_acc": in_1k_acc,
            "in_w_gap": in_w_gap,
            "carton_gap": carton_gap,
            "sin_gap": sin_gap,
            "in_r_gap": in_r_gap,
            "in_9_gap": in_9_gap,
        }

        # evaluate the reliance on color and texture shortcuts on IN-Sketch
        if args.eval_in_sketch:
            in_sketch_acc = self._eval_model(
                dataset_to_loader(self.dataset_dict["in_sketch"]),
                model_name,
                "IN-Sketch",
            )
            in_sketch_gap = in_sketch_acc - in_1k_acc
            log_dict["in_sketch_gap"] = in_sketch_gap

        for key, val in log_dict.items():
            print(f"{key}: {val:.4f}")

        self.log_to_wandb(log_dict)

    @torch.no_grad()
    def _eval_model(self, loader, model_name, dataset_name):
        args = self.args

        total_pred = []
        total_label = []

        self.model.eval()
        pbar = tqdm(
            loader,
            dynamic_ncols=True,
            desc=f"eval {model_name} on {dataset_name}",
        )
        for data_dict in pbar:
            # for data_dict in loader:
            image, target = data_dict["image"], data_dict["target"]
            image = image.to(self.device, non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.model(image)

            if model_name == "seer_rg32gf_ft":
                output = output.logits
            pred = output.argmax(dim=1)
            total_pred.append(pred.cpu())
            total_label.append(target)

        total_pred = torch.cat(total_pred, dim=0)
        total_label = torch.cat(total_label, dim=0)

        total_pred = total_pred.numpy()
        total_label = total_label.numpy()
        acc = ((total_pred == total_label).sum() / len(total_label)).item()

        if dataset_name in ["IN-1k", "IN-W"]:
            carton_class_mask = total_label == CARTON_CLASS_INDEX
            carton_pred = total_pred[carton_class_mask]
            carton_acc = (carton_pred == CARTON_CLASS_INDEX).mean()
            return acc, carton_acc
        else:
            return acc

    def _load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["classifier"])

    def checkpoint(self):
        new_args = copy.deepcopy(self.args)
        training_callable = self.__class__(new_args)
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable)

    def log_to_wandb(self, log_dict):
        if self.args.wandb:
            wandb.log(log_dict, step=0)


def parse_args():
    parser = argparse.ArgumentParser(parents=[slurm_wandb_argparser()])
    parser.add_argument(
        "--model_list",
        type=str,
        nargs="+",
        default=[
            "resnet50",
            "mocov3_r50_lp",
            "style_transfer",
            "mixup",
            "cutmix",
            "cutout",
            "augmix",
            "sd_r50_e2e",
            "lff_r50_e2e",
            "jtt_r50_e2e",
            "eiil_r50_e2e",
            "debian_r50_e2e",
            "resnetv2_50x1_bitm",
            "regnet_y_32gf",
            "seer_rg32gf_ft",
            "swag_rg32gf_lp",
            "swag_rg32gf_ft",
            "vit_b_32",
            "uniform_soup_vit-b",
            "greedy_soup_vit-b",
            "vit_b_16",
            "robust_vit",
            "mocov3_vit-b_lp",
            "mae_vit-b_ft",
            "swag_vit-b_lp",
            "swag_vit-b_ft",
            "vit_l_16",
            "mae_vit-l_ft",
            "swag_vit-l_lp",
            "swag_vit-l_ft",
            "clip_ViT-L-14-336:openai",
            "clip_ViT-L-14:laion400m_e32",
            "mae_vit-h_ft",
            "swag_vit-h_lp",
            "swag_vit-h_ft",
            "clip_ViT-H-14:laion2b_s32b_b79k",
            "clip_ViT-g-14:laion2b_s12b_b42k",
        ],
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval_in_sketch", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()

    if args.wandb:
        assert args.wandb_project_name is not None
        assert args.wandb_entity is not None

    return args


def main():
    args = parse_args()
    args_list = []
    for model in args.model_list:
        cur_args = copy.deepcopy(args)
        cur_args.model = model
        args_list.append(cur_args)

    if args.slurm_partition is not None:
        if not os.path.exists(args.slurm_log_dir):
            os.mkdir(args.slurm_log_dir)

        executor = submitit.AutoExecutor(folder=args.slurm_log_dir)
        executor.update_parameters(
            timeout_min=3 * 24 * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=1,
            cpus_per_task=max(args.num_workers, 1),
            mem_gb=64,
            name=args.slurm_job_name,
            slurm_constraint=args.slurm_constraint,
        )

        job_list = []
        with executor.batch():
            for job_args in args_list:
                trainer = MultiShortcutEvaluator(job_args)
                job = executor.submit(trainer)
                job_list.append(job)

        for job in job_list:
            print("job id: ", job.job_id)
        output_list = [job.result() for job in job_list]
    else:
        for job_args in args_list:
            trainer = MultiShortcutEvaluator(job_args)
            trainer()


if __name__ == "__main__":
    main()
