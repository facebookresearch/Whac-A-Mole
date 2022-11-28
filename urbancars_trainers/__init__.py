"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from .erm import ERMTrainer
from .lff import LfFTrainer
from .spectral_decouple import SpectralDecoupleTrainer
from .groupdro import GroupDROTrainer
from .eiil import EIILTrainer
from .domain_independent import DomainIndependentTrainer
from .jtt import JTTTrainer
from .debian import DebiANTrainer
from .subg import SUBGTrainer
from .mixup import MixupTrainer
from .cutmix import CutMixTrainer
from .cutout import CutoutTrainer
from .lle import LLETrainer
from .cf_f_aug import CFFAugTrainer
from .augmix import AugMixTrainer
from .dfr import DFRTrainer


method_to_trainer = {
    "erm": ERMTrainer,
    "lff": LfFTrainer,
    "sd": SpectralDecoupleTrainer,
    "groupdro": GroupDROTrainer,
    "eiil": EIILTrainer,
    "di": DomainIndependentTrainer,
    "jtt": JTTTrainer,
    "debian": DebiANTrainer,
    "subg": SUBGTrainer,
    "lle": LLETrainer,
    "cf_f_aug": CFFAugTrainer,
    "augmix": AugMixTrainer,
    "mixup": MixupTrainer,
    "cutmix": CutMixTrainer,
    "cutout": CutoutTrainer,
    "dfr": DFRTrainer,
}
