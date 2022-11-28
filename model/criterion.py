"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from LfF:
# https://github.com/alinlab/LfF
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCECriterion(nn.Module):
    def __init__(self, q=0.7, reduction="none"):
        super(GeneralizedCECriterion, self).__init__()
        self.q = q
        self.is_mean_loss = reduction == "mean"

    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight

        if self.is_mean_loss:
            loss = torch.mean(loss)

        return loss
