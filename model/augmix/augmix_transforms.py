"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from AugMix:
# https://github.com/google-research/augmix
# --------------------------------------------------------

import torch
import numpy as np
import model.augmix.augmix_augmentations_pool as augmix_augmentations
import model.augmix.augmix_augmentations_pool_urbancars as augmix_augmentations_urbancars


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, dataset_name="imagenet"):
        self.dataset = dataset
        self.preprocess = preprocess

        if dataset_name == "imagenet":
            self.rand_aug_func = random_aug
        elif dataset_name == "urbancars":
            self.rand_aug_func = random_aug_urbancars
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        data_dict = self.dataset[i]
        x = data_dict["image"]
        x, aug1, aug2 = (
            self.preprocess(x),
            self.rand_aug_func(x, self.preprocess),
            self.rand_aug_func(x, self.preprocess),
        )
        data_dict["image"] = x
        data_dict["aug1"] = aug1
        data_dict["aug2"] = aug2
        return data_dict

    def __len__(self):
        return len(self.dataset)


def random_aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmix_augmentations.augmentations

    mixture_width = 3
    # mixture_depth = -1
    aug_prob_coeff = 1.0
    aug_severity = 1.0

    ws = np.float32(
        np.random.dirichlet([aug_prob_coeff] * mixture_width)
    )
    m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


def random_aug_urbancars(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmix_augmentations_urbancars.augmentations

    mixture_width = 3
    # mixture_depth = -1
    aug_prob_coeff = 1.0
    aug_severity = 1.0

    ws = np.float32(
        np.random.dirichlet([aug_prob_coeff] * mixture_width)
    )
    m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed
