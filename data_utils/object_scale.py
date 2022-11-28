"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F


def crop_mask_and_img(mask: torch.Tensor, image: torch.Tensor):
    """
    Given a mask predicted by a saliency model,
    return the enclosed mask, i.e., cropping the original mask
    by based on the bounding box of the mask.

    Args:
        mask (torch.Tensor): shape: [1, H, W]
        image (torch.Tensor): shape: [3, H, W]

    Returns:
        cropped_mask: shape: [1, h, w], where h <= H, w <= W
        cropped_img: shape: [3, h, w], where h <= H, w <= W
    """
    assert len(mask.shape) == 3
    assert len(image.shape) == 3
    assert mask.shape[1:] == image.shape[1:]

    height, width = mask.shape[1:]
    mask = (mask > 0.5).long()  # to binary mask
    indices = mask[0].nonzero()
    height_indices = indices[:, 0]
    width_indices = indices[:, 1]
    top = height_indices.min()
    bottom = height_indices.max()
    left = width_indices.min()
    right = width_indices.max()

    cropped_mask = mask[:, top : min(bottom + 1, height), left : min(right + 1, width)]
    cropped_img = image[:, top : min(bottom + 1, height), left : min(right + 1, width)]
    return cropped_mask, cropped_img


def rescale_cropped_mask_and_img_with_variance(
    std: float,
    cropped_mask: torch.Tensor,
    cropped_img: torch.Tensor,
    scale: float,
    target_side_size: int = 256,
):

    min_scale = max(0, scale - 0.05)
    max_scale = min(1, scale + 0.05)
    perturbed_scale = (scale + torch.randn(1) * std).clamp(min_scale, max_scale).item()
    return rescale_cropped_mask_and_img(
        cropped_mask,
        cropped_img,
        perturbed_scale,
        target_side_size=target_side_size,
    )


def rescale_cropped_mask_and_img(
    cropped_mask: torch.Tensor,
    cropped_img: torch.Tensor,
    scale: float,
    target_side_size: int = 256,
    object_center_pos=None,
):
    """
    _summary_

    Args:
        cropped_mask (torch.Tensor): cropped mask
        cropped_img (torch.Tensor): cropped image
        scale (float): ranging from 0 to 1
        target_side_size (int): output side size (S) of the rescaled mask (S, S)
    """
    assert len(cropped_mask.shape) == 3
    assert len(cropped_img.shape) == 3
    assert cropped_mask.shape[1:] == cropped_img.shape[1:]
    assert 0 < scale <= 1

    if object_center_pos is None:
        object_center_pos = (target_side_size // 2, target_side_size // 2)

    longer_side = max(cropped_mask.shape[1:])
    scale_factor = scale * target_side_size / longer_side
    cropped_mask = cropped_mask.unsqueeze(0).float()
    rescaled_mask = F.interpolate(
        cropped_mask, scale_factor=scale_factor, mode="nearest-exact"
    ).squeeze(0)

    cropped_img = cropped_img.unsqueeze(0)
    rescaled_img = F.interpolate(
        cropped_img,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    assert rescaled_mask.shape[1:] == rescaled_img.shape[1:]

    rescaled_height, rescaled_width = rescaled_mask.shape[1:]

    target_mask = torch.zeros(
        (1, target_side_size, target_side_size),
        dtype=torch.long,
        device=cropped_mask.device,
    )

    top = max(0, object_center_pos[0] - rescaled_height // 2)
    left = max(0, object_center_pos[1] - rescaled_width // 2)

    target_mask[
        :,
        top : top + rescaled_height,
        left : left + rescaled_width,
    ] = rescaled_mask

    target_img = torch.zeros(
        (3, target_side_size, target_side_size),
        dtype=torch.float,
        device=cropped_img.device,
    )

    target_img[
        :,
        top : top + rescaled_height,
        left : left + rescaled_width,
    ] = rescaled_img

    return target_mask, target_img


def control_object_scale(
    mask: torch.Tensor,
    image: torch.Tensor,
    scale: float,
    std: float,
    target_side_size: int = 256,
):
    """
    set the scale of the object in the mask

    Args:
        mask (torch.Tensor): shape: [1, H, W]
        image (torch.Tensor): shape: [3, H, W]
        scale (float): ranging from 0 to 1
        std (float): std of the gaussian kernel to perturb to object scale to add randomness
        target_size (int): output side size (S) of the rescaled mask (S, S),
    """
    assert len(mask.shape) == 3
    assert len(image.shape) == 3
    assert mask.shape[1:] == image.shape[1:]
    assert 0 < scale <= 1

    min_scale = max(0, scale - 0.05)
    max_scale = min(1, scale + 0.05)
    perturbed_scale = (scale + torch.randn(1) * std).clamp(min_scale, max_scale).item()

    cropped_mask, cropped_img = crop_mask_and_img(mask, image)
    rescaled_mask, rescaled_img = rescale_cropped_mask_and_img(
        cropped_mask, cropped_img, perturbed_scale, target_side_size
    )
    return rescaled_mask, rescaled_img
