# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

wget https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar -O exp/weights/mocov3_r50_lp.pth

wget https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar -O exp/weights/style_transfer_r50.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/mixup_r50.pth -O exp/weights/mixup_r50.pth

wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar -O exp/weights/mocov3_vit-b_lp.pth

wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth -O exp/weights/mae_vit-b_ft.pth

wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth -O exp/weights/mae_vit-l_ft.pth

wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth -O exp/weights/mae_vit-h_ft.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/debian_r50_e2e.pth -O exp/weights/debian_r50_e2e.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/eiil_r50_e2e.pth -O exp/weights/eiil_r50_e2e.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/lff_r50_e2e.pth -O exp/weights/lff_r50_e2e.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/jtt_r50_e2e.pth -O exp/weights/jtt_r50_e2e.pth

wget https://dl.fbaipublicfiles.com/whac_a_mole/sd_r50_e2e.pth -O exp/weights/sd_r50_e2e.pth
