# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import copy
import json
import random
import string
import sys

import numpy as np
from dataclasses import dataclass, field
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict, Tuple
import torch

sent2_cname = None
# Unsupervised datasets
title_cname = 'title'
sectitle_cname = 'sectitles'
sent0_cname = 'text'
sent1_cname = 'text'


# Data collator
@dataclass
class PassageDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    padding_strategy: Union[bool, str, PaddingStrategy]
    max_length: Optional[int] = None
    q_len: Union[tuple, int] = None  # max length of q, and d_len=cap_qd_len-q_len
    d_len: Optional[int] = None  # cap the max length of d

    def __call__(self, batch_data) -> Dict[str, torch.Tensor]:
        '''
        we tokenize in batch, but deal with q/d separately
        '''
        batch_data = [e for e in batch_data if e and 'queries' in e and len(e['queries']) > 0]
        bs = len(batch_data)
        if bs > 0:
            # (IMPORTANT) pad batch to batch_size to avoid hanging in distributed training
            if bs < self.batch_size:
                batch_data.extend([batch_data[0]] * (self.batch_size - len(batch_data)))
            bs = len(batch_data)
        else:
            print('Empty batch?!')
            return
        # flatten sentences and tokenize q/d separately
        contexts = [e['contexts'] for e in batch_data]
        queries = [e['queries'] for e in batch_data]
        docs = [e['docs'] for e in batch_data]
        # sources = [e['sources'][0] if 'sources' in e else '' for e in batch_data]
        # titles = sorted(sorted([e['titles'][0] if 'titles' in e and e['titles'] and e['titles'][0] else '' for e in batch_data]))
        # print('*' * 50)
        # for t, s in zip(titles, sentences):
        #     # print('\t\t', t)
        #     print('\t\t', s[0])
        max_q_len = self.max_length
        max_d_len = self.max_length
        # q_len/d_len specifies the max length of q: can be a range or max length only
        if self.q_len and len(self.q_len) == 2:
            max_q_len = np.random.randint(self.q_len[0], self.q_len[1])
            max_d_len = self.max_length - max_q_len if self.max_length else None
        elif self.q_len and len(self.q_len) == 1:
            max_q_len = self.q_len[0]
        if self.d_len and len(self.d_len) == 2:
            random_d_len = np.random.randint(self.d_len[0], self.d_len[1])
            max_d_len = min(random_d_len, max_d_len) if max_d_len else random_d_len
        elif self.d_len and len(self.d_len) == 1:
            max_d_len = min(self.d_len[0], max_d_len) if max_d_len else self.d_len[0]

        # print('max_q_len=', max_q_len, 'max_d_len=', max_d_len)
        q_feats = self.tokenizer(
            queries,
            max_length=max_q_len,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        k_feats = self.tokenizer(
            docs,
            max_length=max_d_len,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        chunk_feats = None
        if 'random_chunks' in batch_data[0] and len(batch_data[0]['random_chunks']) > 0:
            flat_chunks = [chunk for e in batch_data for chunk in e['random_chunks']]
            chunk_feats = self.tokenizer(
                flat_chunks,
                max_length=max_q_len,
                truncation=True,
                padding=self.padding_strategy,
                return_tensors="pt",
            )
        data = {
            'queries': q_feats,
            'docs': k_feats,
            'contexts_str': contexts,
            'queries_str': queries,
            'docs_str': docs,
        }
        if chunk_feats:
            data['random_chunks'] = chunk_feats
            data['random_chunks_str'] = flat_chunks
        sent_lens, num_overlap_tokens, num_union_tokens = [], [], []
        for i in range(bs):
            sent_len_i = []
            for attn in [q_feats['attention_mask'], k_feats['attention_mask']]:
                sent_len_i.append(sum(attn[i]))
            sent_lens.append(torch.Tensor(sent_len_i).type(torch.int32))
            q_token_set = set(q_feats['input_ids'][i].numpy())
            k_token_set = set(k_feats['input_ids'][i].numpy())
            for stoken_id in self.tokenizer.all_special_tokens:
                q_token_set.discard(stoken_id)
                k_token_set.discard(stoken_id)
            overlap_set = q_token_set.intersection(k_token_set)
            all_set = q_token_set.union(k_token_set)
            num_overlap_tokens.append(len(overlap_set))
            num_union_tokens.append(len(all_set))
        batch = {}
        batch['data'] = data
        batch['length'] = torch.stack(sent_lens)  # [B,2]
        batch['num_word_query'] = torch.Tensor([len(c.split()) for c in queries]).int()  # [B]
        batch['num_word_doc'] = torch.Tensor([len(c.split()) for c in docs]).int()  # [B]
        batch['num_word_context'] = torch.Tensor([len(c.split()) for c in contexts]).int()  # [B]
        batch['num_token_query'] = batch['length'][:, 0]  # [B]
        batch['num_token_doc'] = batch['length'][:, 1]  # [B]
        batch['num_token_overlap'] = torch.Tensor(num_overlap_tokens).int()  # [B]
        batch['num_token_union'] = torch.Tensor(num_union_tokens).int()  # [B]
        # print('sent0_len=', batch['length'].float().mean(dim=0).tolist()[0],
        #       'sent1_len=', batch['length'].float().mean(dim=0).tolist()[1],
        #       'num_overlap_tokens=', np.mean(num_overlap_tokens),
        #       'num_union_tokens=', np.mean(num_union_tokens))
        # print('sent0_minlen=', batch['length'].float().min(dim=0).values.tolist()[0],
        #       'sent1_minlen=', batch['length'].float().min(dim=0).values.tolist()[1],
        #       'sent0_maxlen=', batch['length'].float().max(dim=0).values.tolist()[0],
        #       'sent1_maxlen=', batch['length'].float().max(dim=0).values.tolist()[1])

        if 'neg_docs' in batch_data[0]:
            docs = [e['neg_docs'] for e in batch_data]
            negk_feats = self.tokenizer(
                docs,
                max_length=max_d_len,
                truncation=True,
                padding=self.padding_strategy,
                return_tensors="pt",
            )
            data['neg_docs'] = negk_feats

        return batch


def _extract_title_v1(text, source):
    title = ''
    if source in ['Wikipedia', 'Pile-CC', 'OpenWebText2']:
        lines = [l for l in text.split('\n') if len(l.strip()) > 0]
        if len(lines) > 0: title = lines[0]
    else:
        lines = [l for l in text.split('\n') if len(l.strip().split()) > 3]
        if len(lines) > 0: title = lines[0]
    title = ' '.join(title.split()[:64])  # truncate titles, no longer than 64 words
    return title


def _extract_title(input_text, source, retain_title=False, min_len=16):
    '''
    if title is not given or not Wiki, we take the heading substring as a pseudo title
    '''
    lines = [l for l in input_text.split('\n') if len(l.strip()) > 3]
    title, text = '', input_text
    if source in ['Wikipedia', 'Pile-CC', 'OpenWebText2', 'HackerNews', 'Enron Emails', 'StackExchange', 'PubMed Abstracts']:
        if len(lines) > 0:
            title = lines[0]
            if not retain_title: text = '\n'.join(lines[1:])
    if source in ['ArXiv', 'PubMed Central']:
        input_text = input_text.strip(string.punctuation + string.whitespace + string.digits)
        if input_text.lower().startswith('abstract'): input_text = input_text[8:]
        if input_text.lower().startswith('introduction'): input_text = input_text[12:]
        if input_text.lower().startswith('background'): input_text = input_text[10:]
        input_text = input_text.strip(string.punctuation + string.whitespace).replace('==', '')
    if source == 'USPTO Backgrounds' and input_text.startswith('1. Field of the Invention'):
        input_text = input_text[25:]
    if not title or len(lines) <= 1 or (source != 'Wikipedia' and len(title.split()) <= 2):
        tokens = input_text.split()
        title = ' '.join(tokens[: min(min_len, len(tokens) // 2)])
        if not retain_title: text = ' '.join(tokens[min(min_len, len(tokens) // 2):])
    # corner case, either one is very short
    if len(title.strip()) < min_len or len(text.strip()) < min_len:
        title, text = input_text, input_text
    title = title.replace('\n', '\t').strip()
    if source == 'Enron Emails': title = title.replace('--', '')
    text = text.strip()
    return title, text


def hfdataset_prepare_features(examples,
                               text_field,
                               max_context_len=512, min_dq_len=1,
                               min_q_len=1, max_q_len=512,
                               min_d_len=1, max_d_len=512,
                               q_del_ratio=0.0, d_del_ratio=0.0,
                               dq_prompt_ratio=0.0,
                               max_phrase_num=-1,
                               aug_special_query=False,
                               num_random_chunk=0,
                               pseudo_query_ratio=0.0,
                               pseudo_query_names={},
                               **config_kwargs
                               ):
    ''' examples only contain one element, if using HF_dataset.set_transform() '''
    contexts, queries, docs, random_chunks = [], [], [], []
    try:
        examples = examples['data'] if 'data' in examples else examples
        texts = examples[text_field]
        num_data = len(texts)
        for i in range(num_data):
            # metadata
            id = examples['id'][i] if 'id' in examples else None
            text = examples['text'][i].encode('utf-8', 'ignore').decode()
            url = examples['url'][i] if 'url' in examples else None

            # context
            text_tokens = text.split()
            if max_context_len > 0:
                context_tokens = crop_sequence(text_tokens, max_len=max_context_len, crop_to_maxlen=True)
            else:
                context_tokens = copy.copy(text_tokens)

            # title (candidate-Q)
            if url and 'wikipedia' in url:
                source = 'Wikipedia'
            elif 'meta' in examples and len(examples['meta']) > 0 and examples['meta'][i] and 'pile_set_name' in examples['meta'][i]:
                source = examples['meta'][i]['pile_set_name']
                if 'wikipedia' in source.lower(): source = 'Wikipedia'
            else:
                source = None
            title = examples['title'][i].encode('utf-8', 'ignore').decode() if 'title' in examples and examples['title'][i] else _extract_title_v1(text, source)

            # phrases (candidate-Q)
            if 'font_phrases' in examples and examples['font_phrases'] and examples['font_phrases'][i]:
                ext_phrases = examples['font_phrases'][i] + examples['anchor_phrases'][i]
                abs_phrases = examples['categories'][i] + examples['seealso'][i]
                all_phrases = ext_phrases + abs_phrases
            else:
                ext_phrases, abs_phrases, all_phrases = [], [], []
            ex_dict = {'title': title, 'all_phrases': all_phrases, 'ext_phrases': ext_phrases, 'abs_phrases': abs_phrases}

            # random-crop context (candidate-Q)
            q_tokens = crop_sequence(context_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
            q_tokens = word_replace(q_tokens, replace_ratio=q_del_ratio, replace_with_mask=False)
            ex_dict['random-crop'] = ' '.join(q_tokens)
            # special_query_keys = [k for k in examples.keys() if k.startswith('output-prompt')]
            # special_query = examples[special_query_keys[0]][0].encode('utf-8', 'ignore').decode() if len(special_query_keys) > 0 and examples[special_query_keys[0]][0] else ''
            if 'outputs'in examples and len(examples['outputs']) > 0:
                special_queries = examples['outputs'][0]
            else:
                special_queries = {}
            ex_dict.update(special_queries)

            # prepare D (random-crop context)
            d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            d_tokens = word_replace(d_tokens, replace_ratio=d_del_ratio, replace_with_mask=False)
            d = ' '.join(d_tokens)

            # prepare Q (randomly select a Q from candidates)
            if isinstance(pseudo_query_names, dict) and len(pseudo_query_names) > 0:
                # if a dict is given (names+probs), sample a query type
                psuedo_query_name = random.choices(list(pseudo_query_names.keys()), list(pseudo_query_names.values()))
                psuedo_query_name = psuedo_query_name[0]
                # print(psuedo_query_name)
            elif isinstance(pseudo_query_names, str):
                # single query type is given
                psuedo_query_name = pseudo_query_names
            else:
                raise NotImplementedError('Debug!')
            if isinstance(ex_dict[psuedo_query_name], list):
                cand_phrases = ex_dict[psuedo_query_name] if len(ex_dict[psuedo_query_name]) > 0 else [ex_dict['title']]
                num_phrase = min(max_phrase_num, len(cand_phrases))
                phrases = random.sample(cand_phrases, random.randint(1, num_phrase))
                random.shuffle(phrases)
                _phrases = []
                for p in phrases:
                    p = p[:p.index('|')] if '|' in p else p
                    p = p.strip('[]')
                    _phrases.append(p)
                # print(len(phrases), num_phrase, len(cand_phrases))
                q_tokens = ', '.join(_phrases).split()
            elif isinstance(ex_dict[psuedo_query_name], str):
                q_tokens = ex_dict[psuedo_query_name].split()
            else:
                raise NotImplementedError(f'Not supported for type={type(ex_dict[psuedo_query_name])}: {ex_dict[psuedo_query_name]}')
            if aug_special_query:
                q_tokens = crop_sequence(q_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=q_del_ratio, replace_with_mask=False)

            q = ' '.join(q_tokens)

            # random chunks
            '''
            chunks = []
            for _ in range(num_random_chunk):
                chunk_tokens = crop_sequence(context_tokens, min_len=4, max_len=16)
                if aug_special_query:
                    chunk_tokens = crop_sequence(chunk_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                    chunk_tokens = word_replace(chunk_tokens, replace_ratio=q_del_ratio, replace_with_mask=False)
                chunks.append(' '.join(chunk_tokens))

            add_prompt = np.random.uniform() < dq_prompt_ratio
            if add_prompt:
                q = '[Q]' + q
                d = '[D]' + d
                chunks = ['[Q]' + c for c in chunks]
            random_chunks.append(chunks)
            '''
            contexts.append(' '.join(context_tokens))
            queries.append(q)
            docs.append(d)
            random_chunks.append([])
            # print('Context: ', len(context_tokens), ' '.join(context_tokens).replace('\n', '\t'))
            # print('Q: ', f'[type={psuedo_query_name}]', len(q_tokens), q.replace('\n', '\t'))
            # print('D: ', len(d_tokens), d.replace('\n', '\t'))
            # print('Chunks: ', f'{[len(c.split()) for c in chunks]}', ' | '.join(chunks))
            # print('Chunks: ', len(chunks), f'{[len(c.split()) for c in chunks]}')
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        raise e
        return {'contexts': [[]], 'queries': [[]], 'docs': [[]], 'random_chunks': [[]]}

    assert len(queries) > 0
    return {'contexts': contexts, 'queries': queries, 'docs': docs, 'random_chunks': random_chunks}


def prepare_wiki4valid_features(examples, tokenizer, max_seq_length, padding_strategy):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    # Avoid "None" fields
    valid_ids = [i for i in range(len(examples[sent0_cname]))
                 if examples[sent0_cname][i] is not None]
    total = len(valid_ids)
    if total == 0:
        # print('No valid text for D/Q in Wikipedia for eval')
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }
    # sent0 is doc, sent1 is query
    contexts, queries, docs = [], [], []
    # random crop 1st sentence as query
    for sid in valid_ids:
        sent = examples[sent0_cname][sid]
        tokens = sent.split()
        q_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_cap=8, crop_to_maxlen=False)
        d_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_cap=8, crop_to_maxlen=False)
        queries.append(' '.join(d_tokens))  # cropped psg as doc
        docs.append(' '.join(q_tokens))  # cropped psg as query
        contexts.append(' '.join(tokens))

    # print(f'[id={examples["id"][-1]}][DOC]', len(d_tokens), sents0[-1])
    # print(f'[id={examples["id"][-1]}][QUERY]', len(q_tokens), sents1[-1])
    # print(f'len(sentences)={len(sentences)}')
    return {'contexts': contexts, 'queries': queries, 'docs': docs, 'random_chunks': [queries]}


def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def word_replace(tokens, replace_ratio=0.0, replace_with_mask=False):
    '''
    if replace_with_mask=True, tokens will be replaced with [MASK] randomly
    else it works as token deletion
    '''
    if replace_ratio <= 0.0:
        return tokens
    if replace_with_mask:
        _tokens = ['[MASK]' if np.random.uniform() < replace_ratio else t for t in tokens]
    else:
        _tokens = [t for t in tokens if np.random.uniform() > replace_ratio]
    return _tokens


def crop_sequence(tokens, max_len, min_len=0,
                  max_cap=sys.maxsize, min_cap=0,
                  crop_to_maxlen=False, return_index=False):
    '''
    if crop_to_maxlen is True, we crop the sequence to max_len, at a random position
        otherwise, the length of the cropped sequence is sampled from [min_len, max_len]
        max_cap/min_cap are absolute length limit
    '''
    # directly return if sequence is shorter than max_len
    if crop_to_maxlen and len(tokens) <= max_len:
        if return_index:
            return tokens, 0, len(tokens)
        else:
            return tokens
    if 0 < max_len <= 1:
        max_len = int(len(tokens) * max_len)
    if 0 < min_len <= 1:
        min_len = int(len(tokens) * min_len)
    min_len = min(len(tokens), max(min_cap, min_len))
    max_len = min(len(tokens), max(max_len, min_len), max_cap)
    if crop_to_maxlen:
        cropped_len = max_len
    else:
        cropped_len = np.random.randint(min_len, max_len + 1)
    start_idx = np.random.randint(0, len(tokens) - cropped_len + 1)
    _tokens = tokens[start_idx: start_idx + cropped_len]

    if return_index:
        return _tokens, start_idx, start_idx + cropped_len
    else:
        return _tokens


@dataclass
class PassageDataCollatorWithPaddingQDTogether:
# class PassageDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    padding_strategy: Union[bool, str, PaddingStrategy]
    max_length: Optional[int] = None
    max_q_tokens: Union[tuple] = None  # used if cap_qd_len==True, it's max length of q, and d_len=cap_qd_len-q_len

    def __call__(self, batch_data) -> Dict[str, torch.Tensor]:
        batch_data = [e for e in batch_data if e and 'sentences' in e and len(e['sentences']) > 0]
        bs = len(batch_data)
        if bs > 0:
            # (IMPORTANT) pad batch to batch_size to avoid hanging in distributed training
            if bs < self.batch_size:
                batch_data.extend([batch_data[0]] * (self.batch_size - len(batch_data)))
            bs = len(batch_data)
        else:
            print('Empty batch?!')
            return
        # flatten sentences and tokenize q/d separately
        sentences = [e['sentences'] for e in batch_data]
        # sources = [e['sources'][0] if 'sources' in e else '' for e in batch_data]
        # titles = sorted(sorted([e['titles'][0] if 'titles' in e and e['titles'] and e['titles'][0] else '' for e in batch_data]))
        # print('*' * 50)
        # for t, s in zip(titles, sentences):
        #     # print('\t\t', t)
        #     print('\t\t', s[0])
        num_sent_per_example = len(sentences[0])
        flat_q_sents = [s for e in sentences for s in [e[0]]]
        flat_d_sents = [s for e in sentences for s in [e[1]]]
        flat_sents = flat_q_sents + flat_d_sents
        feats = self.tokenizer(
            flat_sents,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        # print(max_q_len, max_d_len)
        # print('q_shape=', q_feats['input_ids'].shape)
        # print('k_shape=', k_feats['input_ids'].shape)
        # unflatten sentences and return tensors, order is [Q0, D0, Q1, D1]
        batch = {}
        for key in feats:
            qfeat = feats[key][:bs]
            kfeat = feats[key][bs:]
            batch[key] = [qfeat, kfeat]  # [2,B,L], order is [Q0, D0]
            if key == 'attention_mask':
                sent_lens = []
                for i in range(bs):
                    sent_len_i = []
                    for attn in batch[key]:
                        sent_len_i.append(sum(attn[i]))
                    sent_lens.append(torch.Tensor(sent_len_i).type(torch.int32))
                batch['length'] = torch.stack(sent_lens)  # [B,2/4]
        return batch


def hfdataset_prepare_features_not4wikipsg(examples,
                               text_field,
                               max_context_len=256, min_dq_len=10,
                               min_q_len=0.05, max_q_len=0.5,
                               min_d_len=0.05, max_d_len=0.5,
                               word_del_ratio=0.0,
                               dq_prompt_ratio=0.0,
                               title_as_query_ratio=0.0,
                               query_in_doc=0.0,
                               **config_kwargs
                               ):
    ''' examples only contain one element, if using HF_dataset.set_transform() '''
    try:
        examples = examples['data'] if 'data' in examples else examples
        texts = examples[text_field]
        metas = examples['meta'] if 'meta' in examples else [None] * len(texts)
        titles = examples['title'] if 'title' in examples else [''] * len(texts)
        urls = examples['url'] if 'url' in examples else [''] * len(texts)
        extra_input_key = [k for k in examples.keys() if k.startswith('output-prompt')]
        extra_inputs = examples[extra_input_key[0]] if len(extra_input_key) > 0 else [''] * len(texts)
        # print(f'[{len(texts[0].split())}]', texts)
        # (Q1) sent0 is title;
        # (D1) sent1 is psg.
        # (Q2) sent2 is another psg, or a section title/class label (if applicable);
        # (D2) sent3 is another psg.
        sents0, sents1, sents2, sents3 = [], [], [], []
        sources = []
        for text, title, meta, url, extra_input in zip(texts, titles, metas, urls, extra_inputs):
            if not text: continue
            text = text.encode('utf-8', 'ignore').decode()
            title = title.encode('utf-8', 'ignore').decode() if title else ''
            if extra_input:
                extra_input = extra_input.encode('utf-8', 'ignore').decode()
                title = extra_input
            if url and 'wikipedia' in url:
                source = 'Wikipedia'
            elif meta and 'pile_set_name' in meta:
                source = meta['pile_set_name']
                if 'wikipedia' in source.lower(): source = 'Wikipedia'
            else:
                source = None
            sources.append(source)
            if not title:
                # title, text = _extract_title(text, source, retain_title=np.random.uniform() <= include_title_ratio)
                title = _extract_title_v1(text, source)
            # print('*' * 100)
            # print('[source]', source)
            # print('[title]', title.replace('\n', '\t'))
            # print('[text]', len(text), len(text.split()), text.replace('\n', '\t'))
            text_tokens = text.split()
            if max_context_len > 0:
                context_tokens = crop_sequence(text_tokens, max_len=max_context_len, crop_to_maxlen=True)
            else:
                context_tokens = copy.copy(text_tokens)
            # prepare for Q1/D1
            d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            d_tokens = word_replace(d_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            d = ' '.join(d_tokens)
            if title and title_as_query_ratio and np.random.uniform() < title_as_query_ratio:
                title_tokens = title.split()
                q_tokens = crop_sequence(title_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                # q_tokens = crop_sequence(title_tokens, max_len=max_context_len, min_len=min_q_len, min_cap=min_dq_len, crop_to_maxlen=True)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            elif query_in_doc and np.random.uniform() < query_in_doc:
                q_tokens = crop_sequence(d_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            else:
                q_tokens = crop_sequence(context_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            if np.random.uniform() < dq_prompt_ratio:
                q = '[Q]' + q
                # d = '[D]' + source + '[SEP]' + d if source else '[D]' + d
                d = '[D]' + d
            sents0.append(q)
            sents1.append(d)
            # print('Q1: ', len(q_tokens), q.replace('\n', '\t'))
            # print('D1: ', len(d_tokens), d.replace('\n', '\t'))

            # prepare for Q2/D2, sample another set of Q/D in a new context
            '''
            if source == 'Wikipedia':
                sections = _parse_wiki(text, title)
                q, d = sections[np.random.randint(len(sections))]
                q_tokens, d_tokens = q.split(), d.split()
                if max_context_len > 0:
                    q_tokens = crop_sequence(q.split(), max_len=max_context_len, crop_to_maxlen=True)
                    d_tokens = crop_sequence(d.split(), max_len=max_context_len, crop_to_maxlen=True)
                q_tokens = crop_sequence(q_tokens, max_len=max_d_len, min_len=min_d_len, max_cap=max_context_len, min_cap=min_dq_len)
                d_tokens = crop_sequence(d_tokens, max_len=max_d_len, min_len=min_d_len, max_cap=max_context_len, min_cap=min_dq_len)
            else:
                context_tokens = copy.copy(text_tokens)
                if max_context_len > 0:
                    context_tokens = crop_sequence(text_tokens, max_len=max_context_len, crop_to_maxlen=True)
                q_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
                d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            d_tokens = word_replace(d_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            q = ' '.join(q_tokens)
            d = ' '.join(d_tokens)
            if np.random.uniform() < dq_prompt_ratio:
                q = '[Q]' + q
                # d = '[D]' + source + '[SEP]' + d if source else '[D]' + d
                d = '[D]' + d
            # print('Q2: ', len(q_tokens))
            # print('D2: ', len(d_tokens))
            sents2.append(q)
            sents3.append(d)
            '''
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    sentences = sents0 + sents1
    if len(sentences) == 0 or (not all(len(sents0) == len(s) for s in [sents1])):
    # sentences = sents0 + sents1 + sents2 + sents3
    # if len(sentences) == 0 or (not all(len(sents0) == len(s) for s in [sents1, sents2, sents3])):
        print('No valid text for D/Q')
        print(examples)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    return {'sentences': [sentences], 'sources': [sources], 'titles': [titles]}


def simple_document_prepare_features(examples, tokenizer, max_seq_length, padding_strategy):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    # Avoid "None" fields
    docs = []
    try:
        docs = [json.loads(e) for e in examples['text']]
        docs = [d for d in docs if len(d['sections']) > 0]
    except Exception as e:
        # print('Error in loading text from json')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(docs) == 0:
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }

    total = len(docs)
    # sent0 is doc, sent1 is query
    sents0, sents1 = [], []

    try:
        for doc in docs:
            doc['sections'] = [s for s in doc['sections'] if len(s[0].strip()) > 0]
            if len(doc['sections']) == 0:
                continue
            sent = doc['sections'][0][0]
            tokens = sent.split()
            q_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
            d_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
            sents0.append(' '.join(d_tokens))  # cropped psg as doc
            sents1.append(' '.join(q_tokens))  # cropped psg as query
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(sents0) == 0 or len(sents1) == 0:
        print('No valid text for D/Q')
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    sentences = sents0 + sents1
    sent_features = tokenizer(
        sentences,
        max_length=max_seq_length,
        truncation=True,
        padding=padding_strategy,
    )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]]
                             for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

    return features


def _parse_wiki(text, title):
    sec_title = title
    sections = []
    sec_lines = []
    for l in text.split('\n\n'):
        l = l.strip()
        if l.startswith('See also') or l.startswith('Notes') or l.startswith('References') or l.startswith('Further reading') or l.startswith('External links'):
            break
        if len(sec_lines) > 0 and '\n' in l[:30]:
            sections.append((sec_title, '\n'.join(sec_lines)))
            sec_title = l[: l.index('\n')]
            sec_lines = [l[l.index('\n') + 1:]]
        else:
            sec_lines.append(l)
    sections.append((sec_title, '\n'.join(sec_lines)))
    return sections
