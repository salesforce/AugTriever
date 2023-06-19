# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.
# Code adapted from MoCo (https://github.com/facebookresearch/moco)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import torch
from torch import nn
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.bm25.bm25 import BM25Okapi
from src.utils.model_utils import gather_norm


class MLPLayer(nn.Module):
    """
    Dense layer without bias
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        return x


class MLPBiasLayer(nn.Module):
    """
    Dense layer with bias
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        return x


class MLPBiasNormLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.norm(x)
        return x


class MLPActiveLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        # self.dense = weight_norm(nn.Linear(hidden_size, hidden_size, bias=True))
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ProjectorLayer(nn.Module):
    """
    Advanced dense layers for getting sentence representations over pooled representation.
    Modified based on BarlowTwins but uses LayerNorm
    https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    No bias term in each dense layer. First n-1 layers have layer-norm and ReLU.
    """
    def __init__(self, hidden_size, arch):
        super().__init__()
        sizes = [hidden_size] + list(map(int, arch.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            # layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, features, **kwargs):
        x = self.projector(features)

        return x


class MergerLayer(nn.Module):
    """
    Advanced dense layers for getting sentence representations over pooled representation.
    """
    def __init__(self, merger_type, hidden_size):
        super().__init__()
        sizes = [hidden_size] + list(map(int, merger_type.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.projector = nn.Sequential(*layers)

    def forward(self, features, **kwargs):
        x = self.projector(features)
        return x


class BiEncoder(nn.Module):
    '''
    A BiEncoder is meant for training Bi-Encoder arch retrievers
    It consists of two base encoders (including pooler) and optional projecters/mergers.
    '''
    def __init__(self, moco_config, hf_config):
        super(BiEncoder, self).__init__()
        # initialize q_extract
        self.q_extract = getattr(moco_config, 'q_extract', None)
        self.num_random_chunk = getattr(moco_config, 'num_random_chunk', 0)
        if self.q_extract is not None: assert self.num_random_chunk > 0, 'num_random_chunk must be >0 if q_extract enabled'
        self.q_extract_ratio = getattr(moco_config, 'q_extract_ratio', 0.0)
        if self.q_extract is None:
            self.q_extract_model, self.q_extract_tokenizer = None, None
        elif self.q_extract == 'self-dot':
            pass
        elif self.q_extract == 'bm25':
            # make sure `bm25` is a substring of the path :D
            self.q_extract_model = BM25Okapi()
            self.q_extract_model.load_from_json('/export/home/data/search/wiki/UPR_output/bm25-wikipsg/model.json')
        else:
            self.q_extract_tokenizer = AutoTokenizer.from_pretrained(moco_config.q_extract)
            self.q_extract_model = AutoModelForSeq2SeqLM.from_pretrained(moco_config.q_extract)
            self.q_extract_model.half()

        # initialize projector
        # print('q_proj=', moco_config.q_proj)
        # print('k_proj=', moco_config.k_proj)
        self.projection_size = moco_config.projection_size
        self.q_proj = getattr(moco_config, 'q_proj', 'none')
        self.k_proj = getattr(moco_config, 'k_proj', 'none')
        if self.q_proj and self.q_proj != "none":
            if self.q_proj == "mlp":
                self.q_mlp = MLPLayer(self.projection_size)
            elif self.q_proj == "mlpbias":
                self.q_mlp = MLPBiasLayer(self.projection_size)
            elif self.q_proj == "mlpnorm":
                self.q_mlp = MLPBiasNormLayer(self.projection_size)
            elif self.q_proj == "mlpact":
                self.q_mlp = MLPActiveLayer(self.projection_size)
            elif '-' in self.q_proj or self.q_proj.isdigit():
                self.q_mlp = ProjectorLayer(self.projection_size, self.q_proj)
            else:
                raise NotImplementedError('Unknown q_proj ' + self.q_proj)
            self._init_weights(self.q_mlp)
        else:
            self.q_mlp = None
        if self.k_proj and self.k_proj != "none":
            if self.k_proj == "shared":
                self.k_mlp = self.q_mlp
            elif self.k_proj == "mlp":
                self.k_mlp = MLPLayer(self.projection_size)
            elif self.k_proj == "mlpbias":
                self.k_mlp = MLPBiasLayer(self.projection_size)
            elif self.k_proj == "mlpnorm":
                self.k_mlp = MLPBiasNormLayer(self.projection_size)
            elif self.k_proj == "mlpact":
                self.k_mlp = MLPActiveLayer(self.projection_size)
            elif '-' in self.k_proj or self.q_proj.isdigit():
                self.k_mlp = ProjectorLayer(self.projection_size, self.k_proj)
            else:
                raise NotImplementedError('Unknown k_proj ' + self.k_proj)
            self._init_weights(self.k_mlp)
        else:
            self.k_mlp = None

        # initialize merger
        self.merger_type = getattr(moco_config, 'merger_type', None)
        if self.merger_type and '-' in self.merger_type:
            self.merger = MergerLayer(moco_config.merger_type, hf_config.hidden_size)
            self._init_weights(self.merger)
        else:
            self.merger = None

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def state_dict(self):
        state_dict = super(BiEncoder, self).state_dict()
        if 'queue_k' in state_dict: del state_dict['queue_k']
        if 'queue_q' in state_dict: del state_dict['queue_q']
        if 'q_extract_model' in state_dict: del state_dict['q_extract_model']
        if 'q_extract_tokenizer' in state_dict: del state_dict['q_extract_tokenizer']
        return state_dict

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _compute_norm_loss(self, q, k):
        q_norm = gather_norm(q)
        k_norm = gather_norm(k)
        norm_diff = (q_norm - k_norm)**2
        return norm_diff

    def rank_chunks_by_ext_model(self, docs, chunks):
        num_chunk_per_doc = len(chunks) // len(docs)
        doc_encoding = self.q_extract_tokenizer(docs, padding='longest', max_length=192,
                                      truncation=True, add_special_tokens=True, return_tensors='pt')
        chunk_encoding = self.q_extract_tokenizer(chunks, padding='longest', max_length=32,
                                        truncation=True, add_special_tokens=False, return_tensors='pt')
        doc_ids, doc_attention_mask = doc_encoding.input_ids, doc_encoding.attention_mask  # [bs, len]
        chunk_ids, chunk_attention_mask = chunk_encoding.input_ids, chunk_encoding.attention_mask  # [bs*nchunk, len]
        doc_ids = doc_ids.to(self.get_encoder().model.device)
        doc_attention_mask = doc_attention_mask.to(self.get_encoder().model.device)
        chunk_ids = chunk_ids.to(self.get_encoder().model.device)
        chunk_attention_mask = chunk_attention_mask.to(self.get_encoder().model.device)
        doc_ids = torch.repeat_interleave(doc_ids.unsqueeze(1), num_chunk_per_doc, dim=1)  # [bs, len] -> [bs,nchunk,len]
        doc_ids = doc_ids.reshape(-1, doc_ids.shape[-1])  #[bs,nchunk,len] -> [bs*nchunk,len]
        doc_attention_mask = torch.repeat_interleave(doc_attention_mask.unsqueeze(1), num_chunk_per_doc, dim=1)
        doc_attention_mask = doc_attention_mask.reshape(-1, doc_attention_mask.shape[-1])
        with torch.no_grad():
            # print(doc_ids.shape, chunk_ids.shape)
            logits = self.q_extract_model(input_ids=doc_ids, attention_mask=doc_attention_mask, labels=chunk_ids).logits  # [bs,chunk_len,vocab_size]
            # logits = self.model(input_ids=chunk_ids, attention_mask=chunk_attention_mask, labels=doc_ids).logits
            log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)  # [bs,chunk_len,vocab_size]
            nll = -log_softmax.dist_gather(2, chunk_ids.unsqueeze(2)).squeeze(2)  # [bs,chunk_len]
            avg_nll = torch.sum(nll, dim=1)  # [bs*nchunk]
        # self.print_doc_chunks(docs, chunks, avg_nll)
            selected_chunk_ids = avg_nll.reshape(-1, num_chunk_per_doc).argmin(dim=1)
        return selected_chunk_ids

    def print_doc_chunks(self, docs, chunks, avg_nll):
        num_chunk_per_doc = len(chunks) // len(docs)
        for docid, doc in enumerate(docs):
            chunk_scores = []
            doc_chunks = chunks[docid * num_chunk_per_doc: (docid + 1) * num_chunk_per_doc]
            doc_chunk_scores = avg_nll.tolist()[docid * num_chunk_per_doc: (docid + 1) * num_chunk_per_doc]
            print(docid, doc)
            for cid, (chunk, chunk_score) in enumerate(zip(doc_chunks, doc_chunk_scores)):
                item = {
                    "id": cid,
                    "chunk": chunk,
                    "score": chunk_score}
                chunk_scores.append(item)
            chunk_scores = sorted(chunk_scores, key=lambda k:k['score'])
            for item in chunk_scores:
                print('\t[%.2f] %d. %s' % (item['score'], item['id'], item['chunk']))


    def encode(self, sentences, batch_size=32, max_length=512,
               convert_to_numpy: bool = True, convert_to_tensor: bool = False, **kwargs):
        '''
        for MTEB evaluation
        :return:
        '''
        all_embeddings = []
        if convert_to_tensor:
            convert_to_numpy = False

        for start_idx in trange(0, len(sentences), batch_size, desc="docs"):
            documents = sentences[start_idx: start_idx + batch_size]
            inputs = self.tokenizer(documents, max_length=max_length, padding='longest',
                                       truncation=True, add_special_tokens=True,
                                       return_tensors='pt').to(self.encoder_k.model.device)
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            with torch.no_grad():
                batch_weights = self.encoder_k(input_ids, attention_mask)
                if batch_weights.is_cuda:
                    batch_weights = batch_weights.cpu().detach()
                else:
                    batch_weights = batch_weights.detach()
                all_embeddings.extend(batch_weights)

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        return all_embeddings

