# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

method_name=jtt
bias_id_epoch=1

PYTHONPATH=.:$PYTHONPATH python urbancars_trainers/launcher.py \
    --method ${method_name} \
    --amp \
    --num_worker 10 \
    --slurm_job_name ${method_name}_${bias_id_epoch} \
    --bias_id_epoch ${bias_id_epoch} \
    --jtt_up_weight 100 \
    --early_stop_metric_list both \
    --num_seed 6 \
    --wandb
