"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import torch
import numpy as np
import argparse


from torch.utils.data.dataset import Dataset

EPS = 1e-6


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class MultiDimAverageMeter:
    # reference: https://github.com/alinlab/LfF/blob/master/util.py

    def __init__(self, dims):
        self.dims = dims
        self.eye_tsr = torch.eye(dims[0]).long()
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_worst_group_acc(self):
        num_correct = self.cum.reshape(*self.dims)
        cnt = self.cnt.reshape(*self.dims)

        first_shortcut_worst_group_acc = (
            num_correct.sum(dim=2) / cnt.sum(dim=2)
        ).min()
        second_shortcut_worst_group_acc = (
            num_correct.sum(dim=1) / cnt.sum(dim=1)
        ).min()
        both_worst_group_acc = (num_correct / cnt).min()

        return (
            first_shortcut_worst_group_acc,
            second_shortcut_worst_group_acc,
            both_worst_group_acc,
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name} {avg:.3f}"
        return fmtstr.format(**self.__dict__)


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]


class EMAGPU:
    def __init__(self, label, device, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.device = device
        self.parameter = torch.zeros(label.size(0), device=device)
        self.updated = torch.zeros(label.size(0), device=device)
        self.num_class = label.max().item() + 1
        self.max_param_per_class = torch.zeros(self.num_class, device=device)

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

        # update max_param_per_class
        batch_size = len(index)
        buffer = torch.zeros(batch_size, self.num_class, device=self.device)
        buffer[range(batch_size), self.label[index]] = self.parameter[index]
        cur_max = buffer.max(dim=0).values
        global_max = torch.maximum(cur_max, self.max_param_per_class)
        label_set_indices = self.label[index].unique()
        self.max_param_per_class[label_set_indices] = global_max[
            label_set_indices
        ]

    def max_loss(self, label):
        return self.max_param_per_class[label]


def slurm_wandb_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_entity", type=str)

    # SLURM
    parser.add_argument("--slurm_job_name", type=str)
    parser.add_argument("--slurm_constraint", type=str)
    parser.add_argument("--slurm_partition", type=str)
    parser.add_argument("--slurm_mem_gb", type=int, default=128)
    parser.add_argument("--slurm_log_dir", type=str, default="exp/logs")

    return parser
