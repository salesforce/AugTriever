# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import collections
import os
from functools import partial
from typing import List

import datasets
import numpy as np
import torch
from datasets import concatenate_datasets

from src.dataloader.data_configs import load_dataprocess_config
from src.dataloader.data_process import prepare_wiki4valid_features, crop_sequence, word_replace
from src.qa.normalize_text import normalize


BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

def _normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def _create_passage(ctx: dict, normalize: bool=False):
    return BiEncoderPassage(
        _normalize_passage(ctx["text"]) if normalize else ctx["text"],
        ctx["title"],
    )

def _parse_dpr_json(json_sample, exclude_gold=False):
    r = BiEncoderSample()
    r.query = json_sample["question"].replace("’", "'")
    positive_ctxs = json_sample["positive_ctxs"]
    if exclude_gold:
        ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
        if ctxs:
            positive_ctxs = ctxs
    negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
    hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []
    for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
        if "title" not in ctx:
            ctx["title"] = None

    r.positive_passages = [_create_passage(ctx) for ctx in positive_ctxs]
    r.negative_passages = [_create_passage(ctx) for ctx in negative_ctxs]
    r.hard_negative_passages = [_create_passage(ctx) for ctx in hard_negative_ctxs]


# def nq_data_prepare(examples, tokenizer, negative_strategy):
#     num_example = len(examples['question'])
#     sep_token = tokenizer.sep_token if tokenizer.sep_token else '[SEP]'
#     try:
#         # sents0=q+, sents1=d+, sents2=q-, sents3=d-
#         sents0, sents1, sents2, sents3 = [], [], [], []
#         for i in range(num_example):
#             q = examples['question'][i]
#             # ans = examples['answers'][i][0] if 'answers' in examples else ''
#             pos_d = examples['positive_ctxs'][i][0]
#             title = _normalize_passage(pos_d['title']) if 'title' in pos_d else ''
#             text = _normalize_passage(pos_d['text'])
#             d = title + sep_token + text if 'title' in pos_d else text
#             sents0.append(q)
#             sents1.append(d)
#             # prepare a negative example, TODO: add option of sampling among multiple
#             if 'hard_negative_ctxs' in examples and len(examples['hard_negative_ctxs'][i]) > 0:
#                 if negative_strategy == 'first':
#                     neg_d = examples['hard_negative_ctxs'][i][0]
#                 else:
#                     neg_d = random.choice(examples['hard_negative_ctxs'][i])
#             elif 'negative_ctxs' in examples and len(examples['negative_ctxs'][i]) > 0:
#                 neg_d = random.choice(examples['negative_ctxs'][i])
#             else:
#                 neg_d = None
#             title = _normalize_passage(neg_d['title']) if 'title' in neg_d else ''
#             text = _normalize_passage(neg_d['text'])
#             d = title + sep_token + text if 'title' in pos_d and len(title) > 0 else text
#             sents2.append(q)
#             sents3.append(d)
#     except Exception as e:
#         print('Error in processing text to D/Q')
#         print(e)
#         print(examples)
#         return {'sentences': [[]]}
#
#     if not all(len(sents0) == len(s) for s in [sents1, sents2, sents3]):
#         sentences = sents0 + sents1
#     else:
#         sentences = sents0 + sents1 + sents2 + sents3
#
#     return {'sentences': [sentences]}


def dpr_data_prepare(examples,
                     pseudo_query_ratio=0.0,
                     max_context_len=512, min_dq_len=1,
                     min_q_len=1, max_q_len=512,
                     min_d_len=1, max_d_len=512,
                     q_del_ratio=0.0, d_del_ratio=0.0,
                     **kwargs):
    num_example = len(examples['question'])
    try:
        contexts, queries, docs = [], [], []
        for i in range(num_example):
            pos_d = examples['positive_ctxs'][i][0]
            title = normalize(pos_d['title'].strip()) if 'title' in pos_d else ''
            context = normalize(pos_d['text']).split()
            context_tokens = crop_sequence(context, max_len=max_context_len, crop_to_maxlen=True)
            context = ' '.join(context_tokens)
            question = normalize(examples['question'][i])
            d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            d_tokens = word_replace(d_tokens, replace_ratio=d_del_ratio, replace_with_mask=False)
            d = ' '.join(d_tokens)
            if np.random.uniform() <= pseudo_query_ratio:
                q = question
                q_tokens = q.split()
            else:
                q_tokens = crop_sequence(context_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=q_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            contexts.append(context)
            queries.append(q)
            docs.append(d)
            # print('Context: ', len(context_tokens), ' '.join(context_tokens).replace('\n', '\t'))
            # print('Q: ', len(q_tokens), q.replace('\n', '\t'))
            # print('D: ', len(d_tokens), d.replace('\n', '\t'))
            pass
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'queries': [[]], 'docs': [[]]}

    return {'contexts': contexts, 'queries': queries, 'docs': docs}


def load_dpr_dataset(tokenizer, training_args, hftraining_args, moco_args):
    print(training_args.train_file)
    assert os.path.isfile(training_args.train_file), f'{training_args.train_file} does not exist.'
    if training_args.train_file.endswith('.json') or training_args.train_file.endswith('.jsonl'):
        train_dataset = datasets.load_dataset("json", data_files=training_args.train_file,
                                               keep_in_memory=False,
                                               cache_dir=training_args.cache_dir,
                                               streaming=False)
    elif training_args.train_file.endswith('.csv'):
        train_dataset = datasets.load_dataset("csv", data_files=training_args.train_file,
                                               keep_in_memory=False,
                                               cache_dir=training_args.cache_dir,
                                               streaming=False)
    else:
        raise NotImplementedError(f'Not supported file type of data {training_args.train_file}')

    train_dataset = train_dataset['train']

    total_train_batch_size = (
            hftraining_args.train_batch_size
            * hftraining_args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)
            * moco_args.queue_update_steps
    )
    num_examples = total_train_batch_size * (hftraining_args.max_steps + 100)

    # simply duplicate the data multiple times if it needs multiple epochs
    sum_len = len(train_dataset)
    if sum_len < num_examples:
        train_dataset = [train_dataset] * (num_examples // sum_len + 1)
        train_dataset = concatenate_datasets(train_dataset)

    if 'nq' in training_args.train_file:
        parse_fn = partial(nq_data_prepare, tokenizer=tokenizer, dq_prompt_ratio=training_args.dq_prompt_ratio,
                           negative_strategy=training_args.negative_strategy,
                           hard_negative_ratio=training_args.hard_negative_ratio,
                           hard_negative_num=training_args.hard_negative_num)
    else:
        data_prep_config = load_dataprocess_config(training_args, local_rank=hftraining_args.local_rank)
        parse_fn = partial(dpr_data_prepare, **data_prep_config)
    train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)
    train_dataset.set_transform(parse_fn)

    # load a subset of wikipedia as devset
    if training_args.dev_file:
        dev_dataset = datasets.load_dataset("csv",
                                            data_files={"dev": training_args.dev_file},
                                            keep_in_memory=False,
                                            cache_dir=training_args.cache_dir,
                                            delimiter="\t" if "tsv" in training_args.dev_file else ",",
                                            split='dev')
        psg_parse_fn = partial(prepare_wiki4valid_features, tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest')
        dev_dataset.set_transform(psg_parse_fn)
    else:
        dev_dataset = None

    return train_dataset, dev_dataset
