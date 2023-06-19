# Code adapted from Transformers (https://github.com/huggingface/transformers) governed by Apache-2.0 license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import transformers
from transformers.file_utils import ModelOutput
import torch
from torch import nn

from src.model.baseencoder import BaseEncoder

logger = logging.getLogger(__name__)


def mix_two_inputs(input0_tokens, input0_mask, input1_tokens, input1_mask, input0_ratio):
    '''
    Mix input-q to input-k
    :param input0_ratio: ratio of input k
    :return:
    '''
    bsz = len(input1_tokens)
    klen = input0_tokens.shape[1]
    qlen = input1_tokens.shape[1]
    device = input0_tokens.device
    # 1 indicates from input-q, otherwise input-k
    coins = np.random.binomial(size=bsz, p=input0_ratio, n=1)
    q_indices = torch.Tensor([i for i in range(bsz) if not coins[i]]).long().to(device)
    k_indices = torch.Tensor([i for i in range(bsz) if coins[i]]).long().to(device)
    max_len = max(qlen, klen)
    new_k = torch.zeros(bsz, max_len).long().to(device)
    new_k_mask = torch.zeros(bsz, max_len).long().to(device)
    # mask the unselected indices by zero-out
    q_selected = input1_tokens.clone().index_fill_(dim=0, index=k_indices, value=0).long()
    qmask_selected = input1_mask.clone().index_fill_(0, k_indices, 0).long()
    k_selected = input0_tokens.clone().index_fill_(0, q_indices, 0).long()
    kmask_selected = input0_mask.clone().index_fill_(0, q_indices, 0).long()
    # merge unmasked part of q and k
    new_k[q_indices, :qlen] = q_selected[q_indices]
    new_k[k_indices, :klen] = k_selected[k_indices]
    new_k_mask[q_indices, :qlen] = qmask_selected[q_indices]
    new_k_mask[k_indices, :klen] = kmask_selected[k_indices]
    return new_k, new_k_mask


def load_encoder(model_id, pooling=None, hf_config=None):
    if not hf_config:
        hf_config = load_hf(transformers.AutoConfig, model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if model_id.startswith('bert') or model_id.startswith('roberta'):
        # BERT/RoBERTa has default add_pooling_layer=True, a dense layer+tanh activation
        model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_id, config=hf_config,
                                                       add_pooling_layer=False)
    elif model_id.startswith('microsoft/deberta-'):
        # microsoft/deberta-v2-xxlarge, 48 layers, 1536 hidden size, 1.5B parameters
        # microsoft/deberta-v2-xlarge, 24 layers, 1536 hidden size, parameters 900M
        # microsoft/deberta-v3-large, 24 layers, 1024 hidden size, 304M parameters
        # microsoft/deberta-v3-base, 12 layers, 768 hidden size, 86M parameters
        model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_id, config=hf_config)
    else:
        model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_id, config=hf_config)
    retriever = BaseEncoder(tokenizer, model, hf_config, pooling)

    if 'bert' in model_id:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = "[CLS]"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "[SEP]"
    retriever.cls_token_id = tokenizer.cls_token_id

    return retriever, tokenizer


def load_hf(object_class, model_name):
    try:
        obj = object_class.from_pretrained(model_name, local_files_only=True)
    except:
        obj = object_class.from_pretrained(model_name, local_files_only=False)
    return obj


def gather_norm(input, input_mask=None):
    if input_mask is not None:
        _norm = torch.linalg.norm((input * input_mask.unsqueeze(-1)), dim=1)
        _norm = torch.masked_select(_norm, input_mask.bool().reshape(-1))
    else:
        _norm = torch.linalg.norm(input, dim=1, ord=2)
    return _norm.mean()


@dataclass
class ContrastiveLearningOutput(ModelOutput):
    """
    Base class for outputs of sentence contrative learning models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    specific_losses: Optional[dict] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GaussianDropout(nn.Module):
    '''
    Gaussian Dropout from: Fast dropout training
    https://nlp.stanford.edu/pubs/sidaw13fast.pdf
    '''
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x

class VariationalDropout(nn.Module):
    '''
    Variational Dropout from: Variational Dropout and the Local Reparameterization Trick
    https://arxiv.org/pdf/1506.02557.pdf
    '''
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = torch.randn(x.size())
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x