"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import os


from typing import Tuple
from PIL import ImageFont, Image, ImageDraw
from torchvision.transforms.functional import to_tensor, to_pil_image


FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
CARTON_CLASS_INDEX = 478


class AddWatermark:
    def __init__(
        self,
        image_size: int = 224,
        text: str = "捷径捷径捷径",
        font_path: str = os.path.join(FONT_DIR, "SourceHanSerifSC-ExtraLight.otf"),
        opacity: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        x_pos: float = 0.01,
        y_pos: float = 0.4,
    ):
        """
        overlay text class, which helps to reuse the loaded font

        Args:
            image_size (int, optional): size of input image. Defaults to 224.
            font_path (str, optional): _description_. Defaults to SourceHanSerifSC-ExtraLight.
            opacity (float, optional): _description_. Defaults to 1.0.
            color (Tuple[int, int, int], optional): _description_. Defaults to (255, 255, 255).
            x_pos (float, optional): _description_. Defaults to 0.01.
            y_pos (float, optional): _description_. Defaults to 0.4.
        """
        self.color = color
        image_size_to_font_size = {
            224: 36,
            384: 62,
            512: 82,
            518: 84,
        }
        if image_size in image_size_to_font_size:
            font_size = image_size_to_font_size[image_size]
        else:
            font_size = int(image_size / 6.22)

        assert os.path.exists(font_path), f"{font_path} does not exist"
        self.font = ImageFont.truetype(font_path, font_size)
        self.opacity = opacity
        self.x_pos, self.y_pos = x_pos, y_pos
        self.text = text

    def __call__(
        self,
        image: torch.Tensor,
        text: str = None,
    ) -> torch.Tensor:

        if text is None:
            text = self.text

        return _add_watermark(
            image,
            text=text,
            font=self.font,
            opacity=self.opacity,
            color=self.color,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
        )


def _add_watermark(
    image,
    text: str,
    font,
    opacity: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    x_pos: float = 0.0,
    y_pos: float = 0.5,
):
    """
    _summary_

    Args:
        image (torch.Tensor): _description_
        text (str): _description_
        font (_type_): _description_
        opacity (float, optional): _description_. Defaults to 1.0.
        color (Tuple[int, int, int], optional): _description_. Defaults to (255, 255, 255).
        x_pos (float, optional): _description_. Defaults to 0.0.
        y_pos (float, optional): _description_. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    assert (
        0.0 <= opacity <= 1.0
    ), "Opacity must be a value in the range [0.0, 1.0]"
    assert 0.0 <= x_pos <= 1.0, "x_pos must be a value in the range [0.0, 1.0]"
    assert 0.0 <= y_pos <= 1.0, "y_pos must be a value in the range [0.0, 1.0]"

    if isinstance(image, torch.Tensor):
        pil_img = to_pil_image(image)
    else:
        assert isinstance(image, Image.Image)
        pil_img = image

    pil_img = pil_img.convert("RGBA")
    width, height = pil_img.size

    txt = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)

    draw.text(
        xy=(x_pos * width, y_pos * height),
        text=text,
        fill=(color[0], color[1], color[2], round(opacity * 255)),
        font=font,
    )

    output_img = Image.alpha_composite(pil_img, txt).convert("RGB")
    if isinstance(image, torch.Tensor):
        output_img = to_tensor(output_img)

    return output_img
