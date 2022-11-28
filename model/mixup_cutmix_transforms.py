"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from TorchVision:
# https://github.com/pytorch/vision/blob/main/references/classification/train.py
# --------------------------------------------------------

import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
        target_key="label",
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                "Please provide a valid positive value for the num_classes."
                f" Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace
        self.target_key = target_key

    def forward(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if image.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {image.ndim}")
        if label.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {label.ndim}")
        if not image.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {image.dtype}."
            )
        if label.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {label.dtype}"
            )

        if not self.inplace:
            image = image.clone()
            label = label.clone()

        if label.ndim == 1:
            label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            ).to(dtype=image.dtype)

        if torch.rand(1).item() >= self.p:
            return image, label

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image.roll(1, 0)
        target_rolled = label.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        image.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        label.mul_(lambda_param).add_(target_rolled)

        return {"image": image, self.target_key: label}

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            ")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
        target_key="label",
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                "Please provide a valid positive value for the num_classes."
            )
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace
        self.target_key = target_key

    def forward(self, image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if image.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {image.ndim}")
        if label.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {label.ndim}")
        if not image.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {image.dtype}."
            )
        if label.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {label.dtype}"
            )

        if not self.inplace:
            image = image.clone()
            label = label.clone()

        if label.ndim == 1:
            label = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            ).to(dtype=image.dtype)

        if torch.rand(1).item() >= self.p:
            return image, label

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image.roll(1, 0)
        target_rolled = label.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, H, W = F.get_dimensions(image)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        image[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        label.mul_(lambda_param).add_(target_rolled)

        return {"image": image, self.target_key: label}

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            ")"
        )
        return s
