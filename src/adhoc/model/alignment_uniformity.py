# Add Alignment and Uniformity Losses based on https://github.com/ssnl/moco_align_uniform/
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import numpy as np
import torch
import torch.nn.functional as F

def calc_align_alpha2(q, k):
    # used in MoCo-align-uniform, when self.align_alpha == 2
    get_q_bdot_k = (q * k).sum(dim=1)
    loss_align = 2 - 2 * get_q_bdot_k.mean()
    return loss_align

def calc_align_alpha1(q, k):
    # used in MoCo-align-uniform, when self.align_alpha == 1
    return (q - k).norm(dim=1, p=2).mean()

def calc_align(q, k, alpha):
    # defined in https://github.com/SsnL/align_uniform
    return (q - k).norm(p=2, dim=1).pow(alpha).mean()

def calc_uniform(x, t=2):
    # defined in https://github.com/SsnL/align_uniform
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

q = torch.rand(32, 128)
k = torch.rand(32, 128)
q = F.normalize(q, dim=1)
k = F.normalize(k, dim=1)

align_alpha1 = calc_align(q, k, alpha=1)
align_alpha2 = calc_align(q, k, alpha=2)
print(align_alpha1)
print(align_alpha2)

align_moco_alpha1 = calc_align_alpha1(q, k)
align_moco_alpha2 = calc_align_alpha2(q, k)
print(align_moco_alpha1)
print(align_moco_alpha2)

print(calc_uniform(q))
print(calc_uniform(k))