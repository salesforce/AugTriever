# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from functools import partial
import torch

import datasets
from src.dataloader.data_process import prepare_wiki4valid_features
from typing import List, Optional, TypeVar
DatasetType = TypeVar("DatasetType", "Dataset", "IterableDataset")


def prepare_features(examples, hard_negative_num=-1):
    num_example = len(examples['query'])
    try:
        queries, docs, neg_docs = [], [], []
        for i in range(num_example):
            # each medi record (query/pos/neg) is a tuple of (prompt, text)
            _, q = examples['query'][i]
            _, pos_d = examples['pos'][i]
            neg_d = []
            if hard_negative_num > 0:
                _, neg_d = examples['neg'][i]
            queries.append(q)
            docs.append(pos_d)
            neg_docs.extend(neg_d)
            # print('Q: ', len(q.split()), q.replace('\n', '\t'))
            # print('pos D: ', len(pos_d.split()), pos_d.replace('\n', '\t'))
            # if neg_d:
            #     print('neg D: ', len(neg_d.split()), neg_d.replace('\n', '\t'))
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'queries': [[]], 'docs': [[]]}

    if hard_negative_num <= 0: neg_docs = ['']
    return {'contexts': [''], 'queries': queries, 'docs': docs, 'neg_docs': neg_docs}


def load_medi_dataset(tokenizer, training_args, hftraining_args):
    train_dataset, dev_dataset = None, None
    if hftraining_args.do_train and training_args.train_file:
        train_dataset = datasets.load_dataset("json", data_files=[training_args.train_file],
                                              split="train",
                                              keep_in_memory=False, cache_dir=training_args.cache_dir,
                                              streaming=False, ignore_verifications=True)
        total_train_batch_size = (
                hftraining_args.train_batch_size
                * hftraining_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)
        )
        num_examples = total_train_batch_size * (hftraining_args.max_steps + 100)
        sum_len = len(train_dataset)
        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            # print(f"  - Fingerprint_name = {fingerprint_name}")
            print(f"  - Total train_batch_size = {total_train_batch_size}")
            print(f"  \t train_batch_size = {hftraining_args.train_batch_size}")
            print(f"  \t gradient_accumulation_steps = {hftraining_args.gradient_accumulation_steps}")
            print(f"  \t world_size = {(torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)}")
            print(f"  - Total optimization steps = {hftraining_args.max_steps}")
            print(f"  \t max_steps = {hftraining_args.max_steps}")
            print(f"  - Total examples in training sets = {sum_len}")
            print(f"  - Total examples for training = {num_examples}")
            print(f"  - Number of epochs for uniform sampling = {(num_examples // sum_len + 1)}")

        parse_fn = partial(prepare_features, hard_negative_num=training_args.hard_negative_num)
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

    return train_dataset, dev_dataset
