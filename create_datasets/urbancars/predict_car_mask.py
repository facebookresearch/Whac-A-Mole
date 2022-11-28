"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import argparse
import os
import torch.nn.functional as F
import cv2
import scipy.io as sio


from tqdm import tqdm
from create_datasets.urbancars.maskformer.mask_former.config import (
    add_mask_former_config,
)
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config


class MaskPredictor:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_mask_former_config(cfg)
        cfg.merge_from_file(
            "create_datasets/urbancars/metadata/stanford_cars/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml"
        )
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)
        self.pred_seg()

    @torch.no_grad()
    def pred_seg(self):
        args = self.args
        img_root = os.path.join(args.root, f"cars_{args.split}")
        output_split_dir = os.path.join(args.output_dir, f"cars_{args.split}")
        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)
        assert os.path.exists(img_root)

        if args.split == "train":
            _annotations_mat_path = os.path.join(args.root, "devkit", "cars_train_annos.mat")
        else:
            _annotations_mat_path = os.path.join(args.root, "cars_test_annos_withlabels.mat")

        samples = [
            (
                os.path.join(img_root, annotation["fname"]),
                # annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
                [annotation["bbox_x1"], annotation["bbox_y1"], annotation["bbox_x2"], annotation["bbox_y2"]],
            )
            for annotation in sio.loadmat(_annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        for img_fpath, bbox in tqdm(samples):
            img_fname = os.path.basename(img_fpath)
            img_wo_ext = os.path.splitext(img_fname)[0]
            mask_fname = f"{img_wo_ext}_mask.png"
            mask_fpath = os.path.join(output_split_dir, mask_fname)

            if os.path.exists(mask_fpath):
                continue

            img = read_image(img_fpath, format="BGR")
            h, w = img.shape[:2]
            shorter_side = min(h, w)
            if shorter_side > 1024:
                scale = 1024 / shorter_side
                resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                resized_img = img

            outputs = self.predictor(resized_img)

            max_iou_id = None
            max_iou = 0

            panoptic_seg, pred_list = outputs["panoptic_seg"]
            if shorter_side > 1024:
                panoptic_seg = F.interpolate(panoptic_seg.unsqueeze(0).unsqueeze(0).float(), (h, w), mode="nearest")[0, 0].int()

            h, w = panoptic_seg.shape[:2]
            all_one_area = h * w

            x1, y1, x2, y2 = bbox
            bbox_mask = torch.zeros((h, w), dtype=torch.long, device=panoptic_seg.device)
            bbox_mask[y1: y2 + 1, x1: x2 + 1] = 1
            bbox_area = torch.sum(bbox_mask).item()

            for pred in pred_list:
                id = pred["id"]
                if pred["category_id"] not in [2, 7]:
                    continue

                cur_id_mask = panoptic_seg == id
                area = cur_id_mask.sum().item()

                if area == all_one_area:
                    continue

                intersection = (cur_id_mask * bbox_mask).sum().item()
                union = area + bbox_area - intersection
                iou = intersection / (union + 1e-6)

                if iou > max_iou:
                    max_iou = iou
                    max_iou_id = id

            if max_iou_id is None:
                continue

            final_seg = (panoptic_seg == max_iou_id).long()
            np_mask = final_seg.cpu().numpy()
            cv2.imwrite(mask_fpath, np_mask * 255)


def arg_parse():
    parser = argparse.ArgumentParser(description="predict car mask in Stanford Cars")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/stanford_cars/mask",
    )
    parser.add_argument("--root", type=str, default="data/stanford_cars")

    args = parser.parse_args()

    return args


def main():
    args = arg_parse()
    mask_predictor = MaskPredictor(args)
    mask_predictor()


if __name__ == "__main__":
    main()
