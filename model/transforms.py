"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.transforms.functional as tvf
import random
import os


from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop
from imagenet_w.watermark_transform import _add_watermark, FONT_DIR
from typing import List, Tuple
from PIL import ImageFont


class CoRandomHorizontalFlip(RandomHorizontalFlip):
    def forward(self, data):
        img, mask = data
        if torch.rand(1) < self.p:
            return tvf.hflip(img), tvf.hflip(mask)
        return img, mask


class CoRandomResizedCrop(RandomResizedCrop):
    def forward(self, data):
        img, mask = data
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        new_img = tvf.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )
        new_mask = tvf.resized_crop(
            mask, i, j, h, w, self.size, self.interpolation
        )
        return new_img, new_mask


class WatermarkAug:
    def __init__(
        self,
        max_num_chars: int = 8,
        font_file_list: List[str] = [
            os.path.join(FONT_DIR, "SourceHanSerifSC-ExtraLight.otf")
        ],
        font_size_list: List[int] = [32, 36, 40, 44, 48, 52],
        opacity_list: List[float] = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ],
        color_list: List[Tuple[int, int, int]] = [(255, 255, 255)],
        x_pos_list: List[float] = [0.0],
        y_pos_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        assert len(font_file_list) > 0
        assert len(font_size_list) > 0
        assert len(opacity_list) > 0
        assert len(color_list) > 0
        assert len(x_pos_list) > 0
        assert len(y_pos_list) > 0

        self.color_list = color_list
        self.font_list = []
        for font_path in font_file_list:
            for font_size in font_size_list:
                assert os.path.exists(font_path), f"{font_path} does not exist"
                font = ImageFont.truetype(font_path, font_size)
                self.font_list.append(font)
        self.opacity_list = opacity_list
        self.x_pos_list, self.y_pos_list = x_pos_list, y_pos_list
        self.max_num_chars = max_num_chars

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        color = random.choice(self.color_list)
        font = random.choice(self.font_list)
        opacity = random.choice(self.opacity_list)
        x_pos = random.choice(self.x_pos_list)
        y_pos = random.choice(self.y_pos_list)

        # random CJK characters
        text = "".join(
            [
                chr(random.randint(0x4E00, 0x9FFF))
                for _ in range(self.max_num_chars)
            ]
        )

        return _add_watermark(
            image,
            text=text,
            font=font,
            opacity=opacity,
            color=color,
            x_pos=x_pos,
            y_pos=y_pos,
        )
