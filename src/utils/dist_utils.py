# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def dist_gather(x: torch.tensor):
    if not dist.is_initialized():  return x
    x_gather = GatherLayer.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


@torch.no_grad()
def dist_gather_nograd(x: torch.tensor):
    if not dist.is_initialized():  return x
    x_gather = [torch.ones_like(x) for _ in range(get_world_size())]
    dist.all_gather(x_gather, x, async_op=False)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main():
    return get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()
