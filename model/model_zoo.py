"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch
import timm
import clip
import open_clip
import torchvision.transforms as transforms
import torchvision.models as tv_models
import model.mocov3_vits as mocov3_vits
import model.mae_vits as mae_vits


from transformers import RegNetForImageClassification
from model.classifiers import CLIPIN1KZeroShotClassifier
from model.model_soups import get_model_from_sd
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from model.robust_vit.vits import (
    vit_base_patch16_224 as robust_vit_base_patch16_224,
)


def get_model_and_transforms(model_name, device=torch.device(0)):
    if model_name == "resnet50":
        model = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
        )
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "style_transfer":
        model = tv_models.resnet50()
        ckpt_fpath = "exp/weights/style_transfer_r50.pth"
        assert os.path.exists(ckpt_fpath), f"{ckpt_fpath} does not exist"
        state_dict = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
        new_state_dict = {}
        prefix_len = len("module.")
        for k, v in state_dict.items():
            new_state_dict[k[prefix_len:]] = v
        model.load_state_dict(new_state_dict)
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name.startswith("mocov3_"):
        if "r50" in model_name:
            model = tv_models.resnet50()
        elif "vit-b" in model_name:
            model = mocov3_vits.vit_base()
        else:
            raise NotImplementedError
        ckpt_fpath = f"exp/weights/{model_name}.pth"
        checkpoint = torch.load(ckpt_fpath, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        prefix_len = len("module.")
        for k, v in state_dict.items():
            new_state_dict[k[prefix_len:]] = v
        state_dict = new_state_dict

        model.load_state_dict(state_dict)
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "vit_b_16":
        model = tv_models.vit_b_16(
            weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1
        )
        val_transform = tv_models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "vit_b_32":
        model = tv_models.vit_b_32(
            weights=tv_models.ViT_B_32_Weights.IMAGENET1K_V1
        )
        val_transform = tv_models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "vit_l_16":
        model = tv_models.vit_l_16(
            weights=tv_models.ViT_L_16_Weights.IMAGENET1K_V1
        )
        val_transform = tv_models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "swag_vit-l_lp":
        model = tv_models.vit_l_16(
            weights=tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        )
        val_transform = (
            tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        )
    elif model_name == "swag_vit-l_ft":
        model = tv_models.vit_l_16(
            weights=tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        )
        val_transform = (
            tv_models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        )
    elif model_name == "swag_vit-h_lp":
        model = tv_models.vit_h_14(
            weights=tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        )
        val_transform = (
            tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        )
    elif model_name == "swag_vit-h_ft":
        model = tv_models.vit_h_14(
            weights=tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        )
        val_transform = (
            tv_models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        )
    elif model_name.startswith("mae_"):
        arch = model_name.split("_")[1]
        arch_to_vit_arch_name = {
            "vit-b": "vit_base_patch16",
            "vit-l": "vit_large_patch16",
            "vit-h": "vit_huge_patch14",
        }
        vit_arch_name = arch_to_vit_arch_name[arch]
        model = mae_vits.__dict__[vit_arch_name](
            num_classes=1000, global_pool=True
        )
        state_dict = torch.load(f"exp/weights/mae_{arch}_ft.pth")
        model.load_state_dict(state_dict["model"])
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif model_name == "swag_vit-b_lp":
        model = tv_models.vit_b_16(
            weights=tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        )
        val_transform = (
            tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        )
    elif model_name == "swag_vit-b_ft":
        model = tv_models.vit_b_16(
            weights=tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        )
        val_transform = (
            tv_models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        )
    elif model_name == "regnet_y_32gf":
        model = tv_models.regnet_y_32gf(
            weights=tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_V1
        )
        val_transform = (
            tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_V1.transforms()
        )
    elif model_name == "seer_rg32gf_ft":
        model = RegNetForImageClassification.from_pretrained(
            "facebook/regnet-y-320-seer-in1k"
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    384, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif model_name == "swag_rg32gf_lp":
        model = tv_models.regnet_y_32gf(
            weights=tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
        )
        val_transform = (
            tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
        )
    elif model_name == "swag_rg32gf_ft":
        model = tv_models.regnet_y_32gf(
            weights=tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
        )
        val_transform = (
            tv_models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        )
    elif model_name.startswith("clip_"):
        clip_model_name = model_name[len("clip_") :]
        clip_arch, pretrained_dataset = clip_model_name.split(":")
        (clip_model, _, val_transform,) = open_clip.create_model_and_transforms(
            clip_arch, pretrained=pretrained_dataset
        )
        clip_model = clip_model.to(device)
        model = CLIPIN1KZeroShotClassifier(clip_model, device)
    elif model_name in ["uniform_soup_vit-b", "greedy_soup_vit-b"]:
        state_dict = torch.load(f"exp/weights/{model_name}.pth")
        base_model, val_transform = clip.load("ViT-B/32", "cpu", jit=False)
        model = get_model_from_sd(state_dict, base_model)
    elif model_name == "mixup":
        state_dict = torch.load(
            "exp/weights/mixup_r50.pth", map_location="cpu"
        )["state_dict"]
        model = tv_models.resnet50(weights=None)
        model.load_state_dict(state_dict)
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name in [
        "cutmix",
        "cutout",
        "augmix",
    ]:
        model = tv_models.resnet50(weights=None)
        ckpt_fpath = f"exp/weights/{model_name}_r50.pth"
        assert os.path.exists(ckpt_fpath), f"{ckpt_fpath} does not exist"
        state_dict = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
        new_state_dict = {}
        prefix_len = len("module.")
        for k, v in state_dict.items():
            new_state_dict[k[prefix_len:]] = v
        model.load_state_dict(new_state_dict)
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "resnetv2_50x1_bitm":
        model = timm.create_model("resnetv2_101x1_bitm", pretrained=True)
        config = resolve_data_config({}, model=model)
        val_transform = create_transform(**config)
    elif model_name == "robust_vit":
        ckpt_fpath = f"exp/weights/{model_name}.pth"
        state_dict = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
        model = robust_vit_base_patch16_224(pretrained=False)
        model.load_state_dict(state_dict)
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif model_name in [
        "sd_r50_e2e",
        "lff_r50_e2e",
        "jtt_r50_e2e",
        "eiil_r50_e2e",
        "debian_r50_e2e",
    ]:
        state_dict = torch.load(f"exp/weights/{model_name}.pth", map_location="cpu")["state_dict"]
        model = tv_models.resnet50(weights=None)
        model.load_state_dict(state_dict)
        val_transform = tv_models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    else:
        raise NotImplementedError

    model = model.to(device)
    return model, val_transform
