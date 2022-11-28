# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

method_name=erm

PYTHONPATH=.:$PYTHONPATH python urbancars_trainers/launcher.py \
    --method ${method_name} \
    --amp \
    --num_worker 10 \
    --slurm_job_name ${method_name} \
    --early_stop_metric_list both \
    --num_seed 6 \
    --wandb
