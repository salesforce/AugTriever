# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import nn


class BaseEncoder(nn.Module):
    '''
    A wrapper of encoder, with a pooler
    '''
    def __init__(self, tokenizer, model, config, pooling="average", **kwargs):
        super(BaseEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.pooling = pooling
        self.num_view = 0
        self.cls_token_id = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        # append CLS special tokens to input
        # if self.num_view == 1, just use the default CLS
        if self.num_view > 1:
            bs, seqlen = input_ids.shape
            extended_len = seqlen + self.num_view - 1
            extra_cls_tokens = torch.zeros((bs, self.num_view - 1), device=input_ids.device, dtype=input_ids.dtype) + self.cls_token_id
            extra_mask_tokens = torch.ones((bs, self.num_view - 1), device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([extra_cls_tokens, input_ids], dim=1)
            attention_mask = torch.cat([extra_mask_tokens, attention_mask], dim=1)
            position_ids = torch.arange(extended_len, device=input_ids.device).expand(1, -1)

        model_output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        last_hidden = model_output['last_hidden_state']  # [B,L,H]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.)

        if self.pooling.lower() == "average" or self.pooling.lower() == "avg" or self.pooling.lower() == "mean":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  # [B,L,H] -> [B,H]
        elif self.pooling.lower() == "cls":
            emb = last_hidden[:, 0]  # [B,L,H] -> [B,H]
        elif self.pooling.lower() == "multiview":
            emb = last_hidden[:, :self.num_view]  # shape=[B,V,H]
        else:
            raise NotImplementedError('Unknown pooling type:', self.pooling)

        return emb

