# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir data

# preparing Stanford Cars dataset
mkdir data/stanford_cars
mkdir data/stanford_cars/mask
mkdir data/stanford_cars/mask/cars_train
mkdir data/stanford_cars/mask/cars_test

wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz -O data/stanford_cars/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz -O data/stanford_cars/cars_test.tgz
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -O data/stanford_cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat -O data/stanford_cars/cars_test_annos_withlabels.mat
tar xvzf data/stanford_cars/cars_train.tgz -C data/stanford_cars
tar xvzf data/stanford_cars/cars_test.tgz -C data/stanford_cars
tar xvzf data/stanford_cars/car_devkit.tgz -C data/stanford_cars

git clone git@github.com:facebookresearch/MaskFormer.git create_datasets/urbancars/maskformer
mkdir exp
mkdir exp/weights
wget https://dl.fbaipublicfiles.com/maskformer/panoptic-coco/maskformer_panoptic_swin_large_IN21k_384_bs64_554k/model_final_7505c4.pkl -O exp/weights/maskformer_panoptic_swin_large.pkl

PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/predict_car_mask.py --split train
PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/predict_car_mask.py --split test


# preparing LVIS dataset

mkdir data/lvis
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip -O data/lvis/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip -O data/lvis/lvis_v1_val.json.zip
unzip data/lvis/lvis_v1_train.json.zip -d data/lvis
unzip data/lvis/lvis_v1_val.json.zip -d data/lvis


mkdir data/coco
wget http://images.cocodataset.org/zips/train2017.zip -O data/coco/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip

unzip data/coco/train2017.zip -d data/coco
unzip data/coco/val2017.zip -d data/coco


# preparing Places dataset

# Download places with small images (256 * 256)
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar -O data/places365standard_easyformat.tar

tar xvf data/places/places365standard_easyformat.tar -C data
mv data/places365_standard data/places


# generate UrbanCars
mkdir data/urbancars

PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/gen_urbancars.py --split train
PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/gen_urbancars.py --split val
PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/gen_urbancars.py --split test
PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/urbancars_aug_gen_bg_only.py --data_root=data/urbancars/bg-0.95_co_occur_obj-0.95/train
