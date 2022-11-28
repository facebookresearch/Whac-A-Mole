"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .erm import ERMTrainer
from .lff import LfFTrainer
from .sd import SDTrainer
from .eiil import EIILTrainer
from .jtt import JTTTrainer
from .debian import DebiANTrainer
from .wtm_aug import WatermarkAugTrainer
from .bg_aug import BackgroundAugTrainer
from .txt_aug import TextureAugTrainer
from .lle import LLETrainer
from .mixup import MixupTrainer
from .augmix import AugMixTrainer
from .cutmix import CutMixTrainer
from .cutout import CutoutTrainer


method_to_trainer = {
    "erm": ERMTrainer,
    "lff": LfFTrainer,
    "sd": SDTrainer,
    "eiil": EIILTrainer,
    "jtt": JTTTrainer,
    "debian": DebiANTrainer,
    "wtm_aug": WatermarkAugTrainer,
    "bg_aug": BackgroundAugTrainer,
    "txt_aug": TextureAugTrainer,
    "lle": LLETrainer,
    "mixup": MixupTrainer,
    "augmix": AugMixTrainer,
    "cutmix": CutMixTrainer,
    "cutout": CutoutTrainer,
}