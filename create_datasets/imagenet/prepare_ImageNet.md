# Prepare ImageNet and OOD variants of ImageNet Datasets

To evaluate on ImageNet-W (Table 1 in the paper), only ImageNet (more specifically only the validation set) needs to be downloaded. The OOD variants of ImageNet (e.g., ImageNet-R) are only needed when evaluating the reliance on multiple shortcuts (Tables 4, 5 in the paper).

## ImageNet
Download the ImageNet-1k dataset to `data/imagenet`, which needs to contain the `labels.txt` file and two directories for training and validation splits:
```
data/imagenet
├── labels.txt
├── train
│   └── n01440764
│   └── ...
└── val
    ├── n01440764
    └── ...
```
Here `labels.txt` contains 1000 lines, where each line is in the format of `wordnet_ID,class_name`, e.g., *n01440764,tench*.

## Background Augmentation (BG Aug) on ImageNet

We use the saliency detection method developed by (Chay et al.)[http://arxiv.org/abs/2103.12719] to perform background augmentation (BG Aug).

TODO: add the instruction of how to obtain the saliency maps data.

Put the data to `data/imagenet_sal` and the directory tree should look like:
```
data/imagenet_sal
└── train
    └── n01440764
    └── ...
```

Then generate the meta data for background augmentation (BG Aug):
```shell
python create_datasets/imagenet/bg_aug/post_process_imagenet_saliency_detection.py
```

## Stylized ImageNet

Follow the instruction [https://github.com/rgeirhos/Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) to generate the training and validation split of Stylized ImageNet to `data/imagenet-stylized`. The directory tree should look like:
```
data/imagenet-stylized
├── train
│   └── n01440764
│   └── ...
└── val
    ├── n01440764
    └── ...
```

## ImageNet-R

Download the ImageNet-R dataset from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar) and untar it to `data/imagenet-r`. The directory tree should look like:
```
data/imagenet-r
├── n01443537
├── n01484850
├── ...
```

## ImageNet-9 (Background Challenge)

Download the ImagNet-9 (Background Challenge) from [https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz](https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz) and untar it to `data/imagenet_9`. The direction tree should look like this:
```
data/imagenet_9
├── in_to_in9.json
├── mixed_rand
├── mixed_same
├── ...
```

## ImageNet-Sketch

Follow the instruction in (https://github.com/HaohanWang/ImageNet-Sketch)[https://github.com/HaohanWang/ImageNet-Sketch] to download the ImageNet-Sketch dataset and unzip it to `data/imagenet-sketch`. The directory tree should look like:
```
data/imagenet-sketch
├── n01440764
├── n01443537
├── ...
```

## Other OOD variants of ImageNet

### ImageNet-A

Download the ImageNet-R dataset from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar) and untar it to `data/imagenet-a`. The directory tree should look like:
```
data/imagenet-a
├── n01498041
├── n01531178
├── ...
```

### ImageNetV2

Download the matched frequency variant of ImageNetV2 from [https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz) and untar it to `data/imagenetv2/imagenetv2-matched-frequency-format-val`. The directory tree should look like:
```
data/imagenetv2/imagenetv2-matched-frequency-format-val
├── 0
├── 1
├── 2
├── ...
```

### ObjectNet

Follow the instruction in (https://objectnet.dev/download.html)[https://objectnet.dev/download.html] to download ObjectNet. Unzip it to `data/objectnet-1.0` and the directory tree should look like:
```
data/objectnet-1.0
├── LICENSE
├── README
├── images
└── mappings
```

### ImageNet-D

Follow the instruction in (imagenet_d)[https://github.com/bethgelab/robustness/tree/main/examples/imagenet_d] to create ImageNet-D, which requires downloading the source dataset from (here)[http://ai.bu.edu/M3SDA/#dataset] and create the mappings by (map_files.py)[https://github.com/bethgelab/robustness/blob/main/examples/imagenet_d/map_files.py]. Move the folder of ImageNet-D to `data/imagenet-d` and the directory tree should look like:
```
data/imagenet-d
├── clipart
├── infograph
├── painting
├── quickdraw
├── real
└── sketch
```
