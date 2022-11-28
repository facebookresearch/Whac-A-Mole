# Prepare Checkpoint for Shortcut Evaluation


`cd` to repo's root and use [dl_model_ckpt.sh](scripts/prepare_dataset_models/dl_model_ckpt.sh) to download most of the checkpoints.

The checkpoints of other models need to be downloaded manually. Rename files and put them under `exp/weights`
| model        | link                                                                              | filename under `exp/weights` |
|--------------|-----------------------------------------------------------------------------------|------------------------------|
| [CutMix](https://github.com/clovaai/CutMix-PyTorch)       | [link](https://www.dropbox.com/sh/w8dvfgdc3eirivf/AABnGcTO9wao9xVGWwqsXRala?dl=0) | cutmix_r50.pth               |
| [Cutout](https://github.com/clovaai/CutMix-PyTorch)       | [link](https://www.dropbox.com/sh/ln8zk2z7zt2h1en/AAA7z8xTBlzz7Ofbd5L7oTnTa?dl=0) | cutout_r50.pth               |
| [AugMix](https://github.com/google-research/augmix)       | [link](https://drive.google.com/u/0/uc?id=1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF)      | augmix_r50.pth               |
| [Uniform Soup](https://github.com/mlfoundations/model-soups) | [link](https://drive.google.com/u/0/uc?id=1W4QvXCj2E3E6hd9u15861crlzqNTX2cJ)      | uniform_soup_vit-b.pth       |
| [RobustViT](https://github.com/hila-chefer/RobustViT)    | [link](https://drive.google.com/u/1/uc?id=1vDmuvbdLbYVAqWz6yVM4vT1Wdzt8KV-g)      | robust_vit.pth               |

Finally, for Greedy Soup, follow [model-soups](https://github.com/mlfoundations/model-soups) to generate greedy soup checkpoint and move it to `exp/weights/greedy_soup_vit-b.pth`.
