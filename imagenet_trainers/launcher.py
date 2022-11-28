"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import submitit
import os


from imagenet_trainers import method_to_trainer
from utils import slurm_wandb_argparser


def parse_args():
    parser = argparse.ArgumentParser(parents=[slurm_wandb_argparser()])
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "erm",
            "lff",
            "eiil",
            "sd",
            "jtt",
            "debian",
            "wtm_aug",
            "bg_aug",
            "txt_aug",
            "lle",
            "mixup",
            "augmix",
            "cutmix",
            "cutout",
        ],
        required=True,
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="resnet50_erm",
        choices=[
            "resnet50_erm",
            "vit-b_mae-ft",
            "vit-l_mae-ft",
            "vit-h_mae-ft",
            "vit-b_swag-ft",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--epoch", default=90, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=["sgd", "lars"]
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="turn on evaluate mode: only evaluate model on validation set",
    )
    parser.add_argument("--exp_root", type=str, default="exp/imagenet")
    parser.add_argument("--resume", type=str)

    # EIIL
    parser.add_argument("--eiil_n_steps", type=int, default=10000)
    parser.add_argument("--eiil_id_epoch", type=int, default=1)

    # Gradient Starvation
    parser.add_argument(
        "--sp",
        type=float,
        default=1e-4,
        help="coefficient of logits norm penalty in spectral decoupling",
    )

    # JTT
    parser.add_argument("--jtt_id_epoch", type=int, default=1)
    parser.add_argument("--jtt_up_weight", type=int, default=100)

    # GroupDRO
    parser.add_argument("--groupdro_robust_step_size", type=float, default=0.01)
    parser.add_argument("--groupdro_gamma", type=float, default=0.1)

    # LLE
    parser.add_argument(
        "--num_dist_shift", type=int, default=4
    )  # Original, Texture, BG, Watermark
    parser.add_argument("--edge_aug", action="store_true")

    # Mixup
    parser.add_argument("--mixup_alpha", type=float, default=0.2)

    # CutMix
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)

    # cutout
    parser.add_argument("--cutout", type=float, default=0.1)

    parser.add_argument(
        "--eval_in_sketch", action="store_true", help="eval on ImageNet-Sketch"
    )
    parser.add_argument(
        "--eval_other_ood",
        action="store_true",
        help="eval on IN-A, INV2, ObjectNet, and IN-D",
    )

    args = parser.parse_args()

    if args.edge_aug:
        args.num_dist_shift += 1
        print(
            f"set num_dist_shift to {args.num_dist_shift} since using Edge Aug"
        )

    return args


def main():
    args = parse_args()
    Trainer = method_to_trainer[args.method]

    if args.feature_extractor.endswith("_mae-ft"):
        args.optimizer = "lars"
        args.weight_decay = 0.0
    elif args.feature_extractor in ["resnet50_erm", "vit-b_swag-ft"]:
        args.optimizer = "sgd"
    else:
        raise NotImplementedError

    if args.slurm_partition is not None:
        if not os.path.exists(args.slurm_log_dir):
            os.makedirs(args.slurm_log_dir)

        executor = submitit.AutoExecutor(
            folder=args.slurm_log_dir,
            slurm_max_num_timeout=6,
        )
        executor.update_parameters(
            timeout_min=3 * 24 * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=1,
            cpus_per_task=max(args.num_workers, 1),
            mem_gb=args.slurm_mem_gb,
            name=args.slurm_job_name,
            slurm_constraint=args.slurm_constraint,
        )

        trainer = Trainer(args)
        job = executor.submit(trainer)
        print("job id: ", job.job_id)
        print(job.result())
    else:
        trainer = Trainer(args)
        trainer()


if __name__ == "__main__":
    main()
