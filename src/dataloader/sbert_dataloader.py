# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import copy
import random
from functools import partial
import torch
import datasets

from src.dataloader.data_configs import load_dataprocess_config
from src.dataloader.data_process import prepare_wiki4valid_features, hfdataset_prepare_features, crop_sequence, \
    word_replace
from typing import List, Optional, TypeVar
DatasetType = TypeVar("DatasetType", "Dataset", "IterableDataset")

def sbert_prepare_features(examples,
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
            title = examples['title'][i].encode('utf-8', 'ignore').decode()

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


def load_sbert_dataset(tokenizer, training_args, hftraining_args):
    train_dataset, dev_dataset = None, None
    if hftraining_args.do_train:
        # https://huggingface.co/datasets/sentence-transformers/embedding-training-data#available-datasets
        train_dataset = datasets.load_dataset("sentence-transformers/embedding-training-data",
                                               keep_in_memory=False, cache_dir=training_args.cache_dir, streaming=False,
                                               ignore_verifications=True)
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

        data_prep_config = load_dataprocess_config(training_args, local_rank=hftraining_args.local_rank)
        parse_fn = partial(sbert_prepare_features, **data_prep_config)
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
