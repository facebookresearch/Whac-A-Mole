# Edge Augmentation (Edge Aug)

We use edge detection as augmentation (i.e., Edge Aug) to furthur improve the performance of mitigating the color and texture shortcuts on ImageNet-Sketch.

## Use Edge Aug on ImageNet

We use [DexiNed](https://github.com/xavysp/DexiNed) for edge detection on ImageNet. To generate the edge augmenated ImageNet, first download the [checkpoint](https://drive.google.com/u/1/uc?id=1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu&export=download) of DexiNed to `exp/weights/dexined.pth`. Then, run the following command:
```
python create_datasets/imagenet/edge_aug/gen_edge_aug.py
```
It will generate the Edge Aug data to `data/imagenet_edge` with the following directory tree structure:
```
data/imagenet_edge/train
├── n01440764
├── n01443537
├── ...
```


## Training  LLE + Edge Aug

```shell
PYTHONPATH=.:$PYTHONPATH python imagenet_trainers/launcher.py --method ${METHOD} --amp --feature_extractor resnet50_erm --lr ${LR} [--wandb] [--slurm_partition ${SLURM_PARTITION}] [--slurm_job_name ${METHOD}_imagenet] --edge_aug
```

Here are the results and the checkpoints of LLE using Edge Aug:

| method                | architecture | IN-1k | IN-W Gap | Carton Gap | SIN Gap | IN-R Gap | IN-9 Gap | IN-Sketch Gap | LR   | checkpoint |
|-----------------------|--------------|-------|----------|------------|---------|----------|----------|---------------|------|------------|
| LLE + Edge Aug        | ResNet-50    | 76.24 | -6.18    | +10        | -61.52  | -53.69   | -3.95    | -48.25        | 1e-3 | [model](https://dl.fbaipublicfiles.com/whac_a_mole/llr/lle_edge_aug_r50.pth) |
| MAE + LLE + Edge Aug  | ViT-B        | 83.69 | -2.54    | +6         | -59.04  | -43.97   | -3.70    | -43.17        | 1e-3 | [model](https://dl.fbaipublicfiles.com/whac_a_mole/llr/lle_edge_aug_vit-b_mae-ft.pth) |
| MAE + LLE + Edge Aug  | ViT-L        | 85.84 | -1.76    | +16        | -56.52  | -33.76   | -2.94    | -36.45        | 1e-3 | [model](https://dl.fbaipublicfiles.com/whac_a_mole/llr/lle_edge_aug_vit-l_mae-ft.pth) |
| MAE + LLE + Edge Aug  | ViT-H        | 86.84 | -1.20    | +28        | -55.90  | -30.31   | -2.47    | -33.45        | 1e-3 | [model](https://dl.fbaipublicfiles.com/whac_a_mole/llr/lle_edge_aug_vit-h_mae-ft.pth) |
| SWAG + LLE + Edge Aug | ViT-B        | 85.31 | -2.48    | +12        | -61.24  | -27.78   | -3.28    | -38.37        | 1e-4 | [model](https://dl.fbaipublicfiles.com/whac_a_mole/llr/lle_edge_aug_vit-b_swag-ft.pth) |
