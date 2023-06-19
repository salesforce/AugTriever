# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.
# Code adapted from MoCo (https://github.com/facebookresearch/moco)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
import logging

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.model.biencoder import BiEncoder
from src.utils import dist_utils
from src.utils.model_utils import load_encoder, gather_norm, ContrastiveLearningOutput, mix_two_inputs

logger = logging.getLogger(__name__)


class InBatch(BiEncoder):
    def __init__(self, moco_config, hf_config=None, load_model=True):
        super(InBatch, self).__init__(moco_config, hf_config)

        self.moco_config = moco_config
        self.hf_config = hf_config
        self.indep_encoder_k = moco_config.indep_encoder_k
        self.neg_names = moco_config.neg_names  # the indices of data for additional negative examples
        self.projection_size = moco_config.projection_size

        self.sim_metric = getattr(moco_config, 'sim_metric', 'dot')
        self.norm_doc = moco_config.norm_doc
        self.norm_query = moco_config.norm_query
        self.label_smoothing = moco_config.label_smoothing

        # initialize model and tokenizer
        if load_model:
            encoder, tokenizer = load_encoder(moco_config.model_name_or_path, pooling=moco_config.pooling, hf_config=hf_config)
            self.tokenizer = tokenizer
            self.encoder_q = encoder
            if self.indep_encoder_k:
                encoder, tokenizer = load_encoder(moco_config.model_name_or_path, pooling=moco_config.pooling, hf_config=hf_config)
                self.encoder_k = encoder
            else:
                self.encoder_k = self.encoder_q
        else:
            self.tokenizer = None
            self.encoder_q = None
            self.encoder_k = None

        # initialize queue
        self.queue_size = moco_config.queue_size
        assert self.queue_size == 0, "queue_size should be 0 for InBatch model"
        self.queue_ptr = None
        self.queue_k = None


    def forward(self,
        input_ids=None,
        attention_mask=None,
        data=None,
        sent_emb=False,
        is_query=False,
        update_kencoder_queue=True,
        report_align_unif=False,
        report_metrics=False,
        **kwargs
    ):
        if sent_emb:
            return self.infer_forward(
                is_query,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            return self.train_forward(
                data=data,
                update_kencoder_queue=update_kencoder_queue,
                report_align_unif=report_align_unif,
                report_metrics=report_metrics
            )


    def train_forward(self, data, log_prefix='', report_metrics=False, **kwargs):
        q_tokens = data['queries']['input_ids']
        q_mask = data['queries']['attention_mask']
        k_tokens = data['docs']['input_ids']
        k_mask = data['docs']['attention_mask']
        bsz = len(q_tokens)

        # print('q_tokens.shape=', q_tokens.shape, '\n')
        # print('k_tokens.shape=', k_tokens.shape, '\n')

        # 1. compute key
        kemb = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask).contiguous()
        if self.norm_doc:
            kemb = nn.functional.normalize(kemb, dim=-1)
        # 2. compute query
        if self.q_extract and 'random_chunks' in data:
            chunk_tokens = data['random_chunks']['input_ids']
            chunk_mask = data['random_chunks']['attention_mask']
            num_chunk = chunk_tokens.shape[0] // bsz
            # print(chunk_tokens.shape)
            if self.q_extract == 'self-dot':
                with torch.no_grad():
                    q_cand = self.encoder_q(chunk_tokens, chunk_mask).reshape(bsz, -1, kemb.shape[1])  # queries: B,num_chunk,H
                    chunk_score = torch.einsum('bch,bh->bc', q_cand, kemb).detach()  # B,num_chunk
                    chunk_idx = torch.argmax(chunk_score, dim=1)  # [B]
            elif self.q_extract == 'bm25':
                chunk_idx = self.q_extract_model.batch_rank_chunks(batch_docs=data['contexts_str'],
                                                                  batch_chunks=[data['random_chunks_str'][
                                                                                i * num_chunk: (i + 1) * num_chunk]
                                                                                for i in range(len(data['docs_str']))])
            elif self.q_extract_model:
                chunk_idx = self.rank_chunks_by_ext_model(docs=data['contexts_str'], chunks=data['random_chunks_str'])
            else:
                raise NotImplementedError
            c_tokens = torch.stack(
                [chunks[cidx.item()] for chunks, cidx in zip(chunk_tokens.reshape(bsz, num_chunk, -1), chunk_idx)])
            c_mask = torch.stack(
                [chunks[cidx.item()] for chunks, cidx in zip(chunk_mask.reshape(bsz, num_chunk, -1), chunk_idx)])
            if self.q_extract_ratio and self.q_extract_ratio < 1.0:
                c_tokens, c_mask = mix_two_inputs(c_tokens, c_mask, q_tokens, q_mask, input0_ratio=self.q_extract_ratio)
            # print(c_tokens.shape)
            qemb = self.encoder_q(input_ids=c_tokens, attention_mask=c_mask)  # queries: B,H
        else:
            qemb = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: B,H

        if self.norm_query:
            qemb = nn.functional.normalize(qemb, dim=-1)

        # 3. apply projectors
        if self.q_mlp:  qemb = self.q_mlp(qemb)
        if self.k_mlp:  kemb = self.k_mlp(kemb)

        # 4. apply predictor (q/k interaction)
        # 5. computer loss
        gathered_kemb = dist_utils.dist_gather(kemb)
        all_kemb = gathered_kemb
        if self.neg_names is not None and len([data[neg_name]['input_ids'] for neg_name in self.neg_names if neg_name in data]) > 0:
            neg_tokens = torch.cat([data[neg_name]['input_ids'] for neg_name in self.neg_names])
            neg_mask = torch.cat([data[neg_name]['attention_mask'] for neg_name in self.neg_names])
            neg_kemb = self.encoder_k(input_ids=neg_tokens, attention_mask=neg_mask).contiguous()
            gathered_neg_kemb = dist_utils.dist_gather(neg_kemb)
            all_kemb = torch.cat([gathered_kemb, gathered_neg_kemb])

        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)
        labels = labels + dist_utils.get_rank() * len(kemb)
        scores = torch.einsum("id,jd->ij", qemb / self.moco_config.temperature, all_kemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # 6. log stats
        log_stats = {}
        if len(log_prefix) > 0:
            log_prefix = log_prefix + "/"
        log_stats[f"{log_prefix}loss"] = loss.item()

        if report_metrics:
            pred_idx = torch.argmax(scores, dim=-1)
            acc = 100 * (pred_idx == labels).float().mean()
            stdq = torch.std(qemb, dim=0).mean().item()
            stdk = torch.std(kemb, dim=0).mean().item()
            stdqueue_k = torch.std(self.queue_k.T, dim=0).mean() if self.queue_k is not None else 0.0
            log_stats[f"{log_prefix}accuracy"] = acc
            log_stats[f"{log_prefix}stdq"] = stdq
            log_stats[f"{log_prefix}stdk"] = stdk
            log_stats[f'{log_prefix}stdqueue_k'] = stdqueue_k

            doc_norm = gather_norm(kemb)
            query_norm = gather_norm(qemb)
            log_stats[f'{log_prefix}doc_norm'] = doc_norm
            log_stats[f'{log_prefix}query_norm'] = query_norm
            log_stats[f'{log_prefix}norm_diff'] = torch.abs(doc_norm - query_norm)
            log_stats[f'{log_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', qemb, kemb).detach().mean()
            log_stats[f'{log_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', qemb, kemb).detach().fill_diagonal_(
                0).sum() / (bsz * bsz - bsz)

            log_stats[f'{log_prefix}queue_ptr'] = self.queue_ptr
            queue_k_norm = gather_norm(self.queue_k.T) if self.queue_k is not None else 0.0
            log_stats[f'{log_prefix}queue_k_norm'] = queue_k_norm
            if self.neg_names is not None and len([data[neg_name]['input_ids'] for neg_name in self.neg_names if neg_name in data]) > 0:
                log_stats[f'{log_prefix}inbatch_hardneg_score'] = torch.einsum('bd,bd->b', qemb, neg_kemb).detach().mean()
                log_stats[f'{log_prefix}across_neg_score'] = torch.einsum('id,jd->ij', qemb, gathered_neg_kemb).detach().mean()

            # compute on each device, only dot-product
            log_stats[f'{log_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', qemb, kemb).detach().mean()
            log_stats[f'{log_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', qemb, kemb).detach().fill_diagonal_(0).sum() / (bsz * bsz - bsz)
            if self.queue_k is not None:
                log_stats[f'{log_prefix}q@queue_neg_score'] = torch.einsum('id,jd->ij', qemb, self.queue_k.T).detach().mean()

        return ContrastiveLearningOutput(
            loss=loss,
            specific_losses=log_stats
        )

    def infer_forward(
        self,
        is_query,
        input_ids=None,
        attention_mask=None,
    ):
        encoder = self.encoder_q
        if self.indep_encoder_k and not is_query:
            encoder = self.encoder_k
        pooler_output = encoder(input_ids, attention_mask=attention_mask)
        if is_query and self.q_mlp:
            pooler_output = self.q_mlp(pooler_output)
        if not is_query and self.k_mlp:
            pooler_output = self.k_mlp(pooler_output)
        if is_query and self.norm_query:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)
        elif not is_query and self.norm_doc:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
        )
