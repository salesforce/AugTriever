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
import copy
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.model.biencoder import BiEncoder
from src.utils import dist_utils
from src.utils.model_utils import load_encoder, gather_norm, ContrastiveLearningOutput, mix_two_inputs

logger = logging.getLogger(__name__)


class MoCo(BiEncoder):
    def __init__(self, moco_config, hf_config):
        super(MoCo, self).__init__(moco_config, hf_config)
        self.moco_config = moco_config
        self.hf_config = hf_config

        self.model_name_or_path = getattr(moco_config, 'model_name_or_path', 'bert-base-uncased')
        self.use_inbatch_negatives = getattr(moco_config, 'use_inbatch_negatives', False)
        self.chunk_query_ratio = getattr(moco_config, 'chunk_query_ratio', 0.0)
        self.num_extra_pos = getattr(moco_config, 'num_extra_pos', 0)
        self.neg_indices = getattr(moco_config, 'neg_indices', None)
        self.queue_size = moco_config.queue_size
        self.q_queue_size = getattr(moco_config, 'q_queue_size', 0)
        self.active_queue_size = moco_config.queue_size
        self.warmup_queue_size_ratio = moco_config.warmup_queue_size_ratio  # probably
        self.queue_update_steps = moco_config.queue_update_steps  # not useful

        self.momentum = moco_config.momentum
        self.temperature = moco_config.temperature
        self.label_smoothing = moco_config.label_smoothing
        self.norm_doc = moco_config.norm_doc
        self.norm_query = moco_config.norm_query
        self.moco_train_mode_encoder_k = moco_config.moco_train_mode_encoder_k  #apply the encoder on keys in train mode

        self.pooling = moco_config.pooling
        self.sim_metric = getattr(moco_config, 'sim_metric', 'dot')
        self.symmetric_loss = getattr(moco_config, 'symmetric_loss', False)
        self.qk_norm_diff_lambda = getattr(moco_config, 'symmetric_loss', 0.0)
        self.cosine = nn.CosineSimilarity(dim=-1)

        # self.num_q_view = moco_config.num_q_view
        # self.num_k_view = moco_config.num_k_view
        encoder, tokenizer = load_encoder(self.model_name_or_path, moco_config.pooling, hf_config)
        self.tokenizer = tokenizer
        self.encoder_q = encoder

        self.indep_encoder_k = getattr(moco_config, 'indep_encoder_k', False)
        if not self.indep_encoder_k:
            # MoCo
            self.encoder_k = copy.deepcopy(encoder)
            for param_q, param_k in zip(self.encoder_q.model.parameters(), self.encoder_k.model.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        else:
            # independent q/k encoder
            encoder, _ = load_encoder(moco_config.model_name_or_path, moco_config.pooling, hf_config)
            self.encoder_k = encoder

        # create the queue
        # update_strategy = ['fifo', 'priority']
        # self.queue_strategy = moco_config.queue_strategy
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if self.queue_size > 0:
            self.register_buffer("queue_k", torch.randn(moco_config.projection_size, self.queue_size))
            self.queue_k = nn.functional.normalize(self.queue_k, dim=0)  # L2 norm
        else:
            self.queue_k = None
        if self.q_queue_size > 0:
            self.register_buffer("queue_q", torch.randn(moco_config.projection_size, self.q_queue_size))
            self.queue_q = nn.functional.normalize(self.queue_q, dim=0)  # L2 norm
        else:
            self.queue_q = None


    def _update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.model.parameters(), self.encoder_k.model.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue):
        # gather keys before updating queue
        keys = dist_utils.dist_gather_nograd(keys.contiguous())  # [B,H] -> [B*n_gpu,H]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.active_queue_size % batch_size == 0, f'batch_size={batch_size}, active_queue_size={self.active_queue_size}'  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T

    def _compute_loss(self, q, k, queue=None):
        bsz = q.shape[0]
        if self.use_inbatch_negatives:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q.device)  # [B]
            labels = labels + dist_utils.get_rank() * len(k)  # positive indices offset=local_rank*B
            logits = self._compute_logits_inbatch(q, k, queue)  # shape=[B,k*n_gpu] or [B,k*n_gpu+Q]
        else:
            assert self.queue_k is not None
            logits = self._compute_logits(q, k)  # shape=[B,1+Q]
            labels = torch.zeros(bsz, dtype=torch.long).cuda()  # shape=[B]
        # contrastive, 1 positive out of Q negatives (in-batch examples are not used)
        logits = logits / self.temperature
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        # print(q_tokens.device, 'q_shape=', q_tokens.shape, 'k_shape=', k_tokens.shape)
        # print('loss=', loss.item(), 'logits.mean=', logits.mean().item())
        return loss, logits, labels

    def _compute_logits_inbatch(self, q, k, queue):
        k = k.contiguous()
        gathered_k = dist_utils.gather(k)  # [B,H] -> [B*n_gpu,H]
        if self.sim_metric == 'dot':
            logits = torch.einsum("ih,jh->ij", q, gathered_k)  # # [B,H] x [B*n_gpu,H] = [B,B*n_gpu]
            if queue is not None:
                l_neg = torch.einsum('bh,hn->bn', [q, queue.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B,B*n_gpu]+[B,Q] = [B,B*n_gpu+Q]
        else:
            # cast to q.shape=[B,1,H], gathered_k.shape=[B*n_device,H] -> [B,B*n_device]
            logits = self.cosine(q.unsqueeze(1), gathered_k)
            if queue is not None:
                # cast to q.shape=[B,1,H], queue.shape=[Q,H] -> [B,Q]
                l_neg = self.cosine(q.unsqueeze(1), queue.T.clone().detach())  # [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B, B*n_device+Q]
        return logits

    def _compute_logits(self, q, k):
        if self.sim_metric == 'dot':
            assert len(q.shape) == len(k.shape), 'shape(k)!=shape(q)'
            l_pos = torch.einsum('nh,nh->n', [q, k]).unsqueeze(-1)  # [B,H],[B,H] -> [B,1]
            l_neg = torch.einsum('nh,hk->nk', [q, self.queue_k.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
            # print('l_pos=', l_pos.shape, 'l_neg=', l_neg.shape)
            # print('logits=', logits.shape)
        elif self.sim_metric == 'cosine':
            l_pos = self.cosine(q, k).unsqueeze(-1)  # [B,1]
            l_neg = self.cosine(q.unsqueeze(1), self.queue_k.T.clone().detach())  # [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
        else:
            raise NotImplementedError('Not supported similarity:', self.sim_metric)
        return logits

    def _compute_logits_multiview(self, q, k):
        raise NotImplementedError()
        assert self.sim_metric == 'dot'
        assert len(q.shape) == len(k.shape), 'shape(k)!=shape(q)'
        # multiple view, q and k are represented with multiple vectors
        bs = q.shape[0]
        emb_dim = q.shape[-1]
        _q = q.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
        _k = k.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
        l_pos = torch.einsum('nc,nc->n', [_q, _k]).unsqueeze(-1)  # [B*V,H],[B*V,H] -> [B*V,1]
        l_pos = l_pos.reshape(bs, -1).max(dim=1)[0]  # [B*V,1] -> [B,V] ->  [B,1]
        # or l_pos=torch.diag(torch.einsum('nvc,mvc->nvm', [q, k]).max(dim=1)[0], 0)
        _queue = self.queue_k.detach().permute(1, 0).reshape(-1, self.num_k_view, emb_dim)  # [H,Q*V] -> [Q*V,H] -> [Q,V,H]
        l_neg = torch.einsum('nvc,mvc->nvm', [q, _queue])  # [B,V,H],[Q,V,H] -> [B,V,Q]
        l_neg = l_neg.reshape(bs, -1).max(dim=1)[0]  # [B,V,Q] -> [B,Q]
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B*V, 1+Q*V]
        logits = logits.reshape(q.shape[0], q.shape[1], -1)  # [B*V, 1+Q*V] -> [B,V,1+Q*V]
        logits = logits.max(dim=1)  # TODO, take rest views as negatives as well?
        return logits


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

    def train_forward(self, data, log_prefix='',
                      update_kencoder_queue=True, report_align_unif=False, report_metrics=False, **kwargs):
        q_tokens = data['queries']['input_ids']
        q_mask = data['queries']['attention_mask']
        k_tokens = data['docs']['input_ids']
        k_mask = data['docs']['attention_mask']

        log_stats = {}
        bsz = q_tokens.size(0)

        # 1. compute key
        if self.indep_encoder_k:
            k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: B,H
        else:
            with torch.no_grad():  # no gradient to keys
                if update_kencoder_queue:
                    self._update_encoder_k()  # update the key encoder
                if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                    self.encoder_k.eval()

                k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: B,H
        if self.norm_doc:
            k = nn.functional.normalize(k, dim=-1)
        # 2. compute query
        if self.q_extract and 'random_chunks' in data:
            # print(chunk_tokens.shape)
            chunk_tokens = data['random_chunks']['input_ids']
            chunk_mask = data['random_chunks']['attention_mask']
            num_chunk = chunk_tokens.shape[0] // bsz
            if self.q_extract == 'self-dot':
                with torch.no_grad():
                    q_cand = self.encoder_q(input_ids=chunk_tokens, attention_mask=chunk_mask).reshape(bsz, -1, k.shape[1])  # queries: B,num_chunk,H
                    chunk_score = torch.einsum('bch,bh->bc', q_cand, k).detach()  # B,num_chunk
                    chunk_idx = torch.argmax(chunk_score, dim=1)  # [B]
            elif self.q_extract == 'bm25':
                chunk_idx = self.q_extract_model.batch_rank_chunks(batch_docs=data['contexts_str'],
                                                                  batch_chunks=[data['random_chunks_str'][i*num_chunk: (i+1)*num_chunk]
                                                                         for i in range(len(data['docs_str']))])
            elif self.q_extract_model:
                chunk_idx = self.rank_chunks_by_ext_model(docs=data['contexts_str'], chunks=data['random_chunks_str'])
            else:
                raise NotImplementedError
            c_tokens = torch.stack([chunks[cidx.item()] for chunks, cidx in zip(chunk_tokens.reshape(bsz, num_chunk, -1), chunk_idx)])
            c_mask = torch.stack([chunks[cidx.item()] for chunks, cidx in zip(chunk_mask.reshape(bsz, num_chunk, -1), chunk_idx)])
            if self.q_extract_ratio is not None:
                c_tokens, c_mask = mix_two_inputs(c_tokens, c_mask, q_tokens, q_mask, input0_ratio=self.q_extract_ratio)
            q = self.encoder_q(input_ids=c_tokens, attention_mask=c_mask)  # queries: B,H
        else:
            q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: B,H
        if self.norm_query: q = nn.functional.normalize(q, dim=-1)

        # 3. apply projectors
        if self.q_mlp:  q = self.q_mlp(q)
        if self.k_mlp:  k = self.k_mlp(k)

        # 4. apply predictor (q/k interaction)
        # 5. computer loss
        loss, logits, labels = self._compute_loss(q, k.detach(), self.queue_k)
        if self.symmetric_loss:
            _loss, _logits, _labels = self._compute_loss(k, q.detach(), self.queue_q)
            loss = (loss + _loss) / 2
        if self.qk_norm_diff_lambda > 0:
            norm_loss = self._compute_norm_loss(q, k)
            loss += self.qk_norm_diff_lambda * norm_loss
        else:
            norm_loss = 0.0
        if len(log_prefix) > 0:
            log_prefix = log_prefix + '/'
        log_stats[f'{log_prefix}cl_loss'] = torch.clone(loss)
        if report_metrics:
            pred_idx = torch.argmax(logits, dim=-1)
            acc = 100 * (pred_idx == labels).float()
            stdq = torch.std(q, dim=0).mean()  # q=[Q,H], std(q)=[H], std(q).mean()=[1]
            stdk = torch.std(k, dim=0).mean()
            stdqueue_k = torch.std(self.queue_k.T, dim=0).mean() if self.queue_k is not None else 0.0
            stdqueue_q = torch.std(self.queue_q.T, dim=0).mean() if self.queue_q is not None else 0.0
            # print(accuracy.detach().cpu().numpy())
            log_stats[f'{log_prefix}accuracy'] = acc.mean()
            log_stats[f'{log_prefix}stdq'] = stdq
            log_stats[f'{log_prefix}stdk'] = stdk
            log_stats[f'{log_prefix}stdqueue_k'] = stdqueue_k
            log_stats[f'{log_prefix}stdqueue_q'] = stdqueue_q

            doc_norm = gather_norm(k)
            query_norm = gather_norm(q)
            log_stats[f'{log_prefix}doc_norm'] = doc_norm
            log_stats[f'{log_prefix}query_norm'] = query_norm
            log_stats[f'{log_prefix}norm_diff'] = torch.abs(doc_norm - query_norm)

            queue_k_norm = gather_norm(self.queue_k.T) if self.queue_k is not None else 0.0
            queue_q_norm = gather_norm(self.queue_q.T) if self.queue_q is not None else 0.0
            log_stats[f'{log_prefix}queue_ptr'] = self.queue_ptr
            log_stats[f'{log_prefix}active_queue_size'] = self.active_queue_size
            log_stats[f'{log_prefix}queue_k_norm'] = queue_k_norm
            log_stats[f'{log_prefix}queue_q_norm'] = queue_q_norm
            log_stats[f'{log_prefix}norm_loss'] = norm_loss

            # compute on each device, only dot-product
            log_stats[f'{log_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', q, k).detach().mean()
            log_stats[f'{log_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', q, k).detach().fill_diagonal_(0).sum() / (bsz * bsz - bsz)
            if self.queue_k is not None:
                log_stats[f'{log_prefix}q@queue_neg_score'] = torch.einsum('id,jd->ij', q, self.queue_k.T).detach().mean()

        # lazily computed & cached!
        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result
        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = (q @ self.queue_k.detach()).flatten()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result
        def get_q_dot_queue_splits():
            # split queue to 4 parts, take a slice from each part, same size to q
            # Q0 is oldest portion, Q3 is latest
            if not hasattr(get_q_dot_queue_splits, 'result'):
                get_q_dot_queue_splits.result = []
                # organize the key queue to make it in the order of oldest -> latest
                queue_old2new = torch.concat([self.queue_k.clone()[:, int(self.queue_ptr): self.active_queue_size],
                                              self.queue_k.clone()[:, :int(self.queue_ptr)]], dim=1)
                for si in range(4):
                    queue_split = queue_old2new[:, self.active_queue_size // 4 * si: self.active_queue_size // 4 * (si + 1)]
                    get_q_dot_queue_splits.result.append(q @ queue_split)
            return get_q_dot_queue_splits.result
        def get_queue_dot_queue():
            if not hasattr(get_queue_dot_queue, 'result'):
                get_queue_dot_queue.result = torch.pdist(self.queue_k.T, p=2)
            assert get_queue_dot_queue.result._version == 0
            return get_queue_dot_queue.result

        log_stats[f'{log_prefix}loss'] = loss

        if update_kencoder_queue:
            # print('Before \t\t queue_ptr', int(self.queue_ptr), 'queue_k.shape=', self.queue_k.shape)
            if self.queue_k is not None: self._dequeue_and_enqueue(k, self.queue_k)
            if self.queue_q is not None: self._dequeue_and_enqueue(q, self.queue_q)
            if self.queue_k is not None or self.queue_q is not None:
                ptr = int(self.queue_ptr)
                ptr = (ptr + bsz * dist_utils.get_world_size()) % self.active_queue_size  # move pointer
                self.queue_ptr[0] = ptr
            # print('After \t\t queue_ptr', int(self.queue_ptr), 'self.queue_k.shape=', self.queue_k.shape)
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
        # self.encoder_k is momentum during training, so be cautious
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
