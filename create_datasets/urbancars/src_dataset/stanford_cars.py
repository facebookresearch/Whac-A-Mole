"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from TorchVision:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/stanford_cars.py
# --------------------------------------------------------

import pathlib
import torch
import os
import json


from typing import Callable, Optional, Any, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it."""

    mask_base_folder = "stanford_cars/mask"
    filter_objs_metadata_base_folder = (
        "create_datasets/urbancars/metadata/stanford_cars"
    )

    def __init__(
        self,
        root: str,
        urban_or_country: int,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_bbox: bool = False,
        return_image_path: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. This dataset needs to have scipy"
                " installed: pip install scipy"
            )

        super().__init__(
            root,
            transform=transform,
        )
        assert urban_or_country in [0, 1]
        self.mask_transform = mask_transform
        self._split = split
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = (
                self._base_folder / "cars_test_annos_withlabels.mat"
            )
            self._images_base_path = self._base_folder / "cars_test"

        self._mask_split_path = os.path.join(
            root, self.mask_base_folder, f"cars_{split}"
        )

        with open(
            os.path.join(self.filter_objs_metadata_base_folder, f"{split}.txt")
        ) as f:
            filtered_filename_list = f.read().splitlines()

        filtered_filename_set = set(filtered_filename_list)

        with open(
            os.path.join(
                self.filter_objs_metadata_base_folder, "body_type.json"
            )
        ) as f:
            urban_or_country_list = json.load(f)

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"]
                - 1,  # Original target mapping  starts from 1, hence -1
                [
                    annotation["bbox_x1"],
                    annotation["bbox_y1"],
                    annotation["bbox_x2"],
                    annotation["bbox_y2"],
                ],
            )
            for annotation in sio.loadmat(
                self._annotations_mat_path, squeeze_me=True
            )["annotations"]
            if annotation["fname"] in filtered_filename_set
            and urban_or_country_list[annotation["class"] - 1]
            == urban_or_country
        ]

        self.classes = sio.loadmat(
            str(devkit / "cars_meta.mat"), squeeze_me=True
        )["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.return_bbox = return_bbox
        self.return_image_path = return_image_path

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target, bbox = self._samples[idx]

        mask_fname = (
            os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        )
        mask_path = os.path.join(self._mask_split_path, mask_fname)

        pil_image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = mask.resize(pil_image.size)

        if self.transform is not None:
            pil_image = self.transform(pil_image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        data_dict = {
            "image": pil_image,
            "target": target,
            "mask": mask,
        }

        if self.return_bbox:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            new_bbox = [x1, y1, w, h]
            new_bbox = torch.tensor(new_bbox, dtype=torch.long)
            data_dict["bbox"] = new_bbox

        if self.return_image_path:
            data_dict["image_path"] = image_path

        return data_dict
