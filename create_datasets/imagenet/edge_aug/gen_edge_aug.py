"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import argparse
import torchvision.transforms as transforms
import os


from tqdm import tqdm
from create_datasets.imagenet.edge_aug.dexined_model import DexiNed
from dataset.imagenet import ImageNet
from torchvision.utils import save_image
from torch.utils.data import default_collate


def in_edge_gen_collate(batch):
    image_list = []
    file_path_list = []

    for data_dict in batch:
        image_list.append(data_dict["image"])
        file_path_list.append(data_dict["file_path"])

    batch_image = default_collate(image_list)

    return {"image": batch_image, "file_path": file_path_list}


class ImageNetEdgeGenerator:
    def __call__(self, args):
        device = torch.device(0)
        self.device = device

        model = DexiNed()
        state_dict = torch.load(args.dexined_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        self.model = model.to(device)

        size = 512
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size=(size, size)),
                transforms.ToTensor(),
                lambda x: x * 255,
                transforms.Normalize(
                    mean=[123.68, 116.779, 103.939], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        train_set = ImageNet(
            "data",
            "train",
            transform=transform,
            return_file_path=True,
        )
        self.loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=80,
            shuffle=False,
            num_workers=80,
            pin_memory=True,
            collate_fn=in_edge_gen_collate,
        )

        self.output_dir = args.in_edge_train_root
        in1k_root = args.in_1k_train_root

        for wn_id in os.listdir(in1k_root):
            os.makedirs(os.path.join(self.output_dir, wn_id), exist_ok=True)

        self.generate()

    @torch.no_grad()
    def generate(self):
        for data_dict in tqdm(self.loader):
            file_path_list = data_dict["file_path"]

            idx_for_gen = []
            file_path_for_gen_list = []
            for idx, file_path in enumerate(file_path_list):
                wn_id, fname = file_path.split("/")[-2:]
                output_fpath = os.path.join(self.output_dir, wn_id, fname)
                if not os.path.exists(output_fpath):
                    idx_for_gen.append(idx)
                    file_path_for_gen_list.append(output_fpath)

            if len(idx_for_gen) == 0:
                continue

            image = data_dict["image"].to(self.device, non_blocking=True)
            image = image[idx_for_gen]
            image = image[:, [2, 1, 0]]  # RGB to BGR
            with torch.cuda.amp.autocast(enabled=True):
                batch_edge = 1 - torch.sigmoid(self.model(image))

            for edge, output_path in zip(batch_edge, file_path_for_gen_list):
                save_image(edge, output_path, nrow=1, padding=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexined_ckpt", type=str, default="exp/weights/dexined.pth")
    parser.add_argument("--in_1k_train_root", type=str, default="data/imagenet/train")
    parser.add_argument("--in_edge_train_root", type=str, default="data/imagenet_edge/train")
    args = parser.parse_args()

    generator = ImageNetEdgeGenerator(args)
    generator()


if __name__ == "__main__":
    main()
