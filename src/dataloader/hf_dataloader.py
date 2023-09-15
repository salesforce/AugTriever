# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os.path
import random
import string
from functools import partial
import torch
import numpy as np
from datetime import datetime

import datasets
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset, _concatenate_map_style_datasets, _interleave_map_style_datasets
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit

from src.dataloader.data_configs import load_dataprocess_config
from src.dataloader.data_process import prepare_wiki4valid_features, hfdataset_prepare_features
from typing import List, Optional, TypeVar
DatasetType = TypeVar("DatasetType", "Dataset", "IterableDataset")


def _interleave_map_style_datasets(
    datasets: List["Dataset"],
    num_step: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    new_fingerprint: str = None,
    keep_in_memory: bool = False,
    **kwargs,
) -> "Dataset":
    """
    Modified based on _interleave_map_style_datasets() in datasets.arrow_dataset.py
    """
    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = _concatenate_map_style_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    start = datetime.now()
    if probabilities is None:
        # Example:: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    elif len(probabilities) == 1:
        indices = np.tile(np.arange(lengths[0]), reps=(num_step // lengths[0])+1)[:num_step].tolist()
    else:
        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=10000000, p=probabilities))
        current_index = [0] * len(datasets)
        indices = []

        for source_idx in iter_random_indices():
            # keep sampling until we have enough number of training indices
            if len(indices) >= num_step:
                break
            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] = (current_index[source_idx] + 1) % lengths[source_idx]
    # end = datetime.now()
    # print('Time used for indices sampling', end - start)
    # print('#indices=', len(indices))

    # start = datetime.now()
    merged_dataset = concatenated_datasets.select(indices, keep_in_memory=False,
                                                  writer_batch_size=10000000,
                                                  new_fingerprint=new_fingerprint,
                                                  **kwargs)
    # end = datetime.now()
    # print('Time used for datasets.select()', end - start)
    return merged_dataset


def interleave_datasets(
    datasets: List[DatasetType],
    num_step: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    new_fingerprint: str = None,
    keep_in_memory: bool = False,
) -> DatasetType:
    """
    Rui:  interleave_datasets can be 2-5x slower and consumes extra memory
    keep_in_memory=False is very slow, keep_in_memory=True results in OOM
    Modified based on interleave_datasets() in datasets.combine.py
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    """
    from datasets.arrow_dataset import Dataset
    map_style = isinstance(datasets[0], Dataset)
    for dataset in datasets[1:]:
        if (map_style and not isinstance(dataset, Dataset)):
            raise ValueError(
                f"Unable to interleave a {type(datasets[0])} with a {type(dataset)}. Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    return _interleave_map_style_datasets(datasets, num_step, probabilities, seed,
                                          info=info, split=split,
                                          new_fingerprint=new_fingerprint,
                                          keep_in_memory=keep_in_memory)


def load_datasets(tokenizer, training_args, hftraining_args, moco_args):
    train_dataset, dev_dataset = None, None
    if hftraining_args.do_train and training_args.train_file:
        # wikipedia is implemented in Apache Beam and it's not streamable
        streaming = False  # if 'wiki' in training_args.train_file else True
        train_dataset_names = training_args.train_file.split(':')
        train_datasets = []
        for dataset_name in train_dataset_names:
            if dataset_name.startswith('beir_'):
                beir_dataset = dataset_name[5:]
                corpus_jsonl_path = os.path.join(training_args.beir_path, beir_dataset, 'corpus.jsonl')
                loaded_dataset = datasets.load_dataset("json", data_files=corpus_jsonl_path,
                                                        keep_in_memory=False, cache_dir=training_args.cache_dir,
                                                        ignore_verifications=True,
                                                        streaming=streaming)
                loaded_dataset = loaded_dataset['train']
                if 'metadata' in loaded_dataset.column_names:
                    loaded_dataset = loaded_dataset.remove_columns('metadata')
                title_field, text_field = 'title', 'text'
            elif dataset_name.startswith('pile_'):
                pile_dataset = dataset_name[5:]
                corpus_jsonl_path = os.path.join('/export/home/data/pretrain/pile/', f'{pile_dataset}.json')
                # corpus_jsonl_path = os.path.join('/export/home/data/pretrain/pile/10k/', f'{pile_dataset}.json')
                loaded_dataset = datasets.load_dataset("json", data_files=corpus_jsonl_path,
                                                       keep_in_memory=False,
                                                       cache_dir=training_args.cache_dir, streaming=streaming,
                                                       ignore_verifications=True,
                                                       num_proc=4)
                loaded_dataset = loaded_dataset['train']
                title_field, text_field = None, 'text'
            elif dataset_name == 'c4':
                # https://huggingface.co/datasets/c4
                # #en=364,868,892, #realnewslike=13,799,838, columns=['url', 'timestamp', 'text']
                loaded_dataset = datasets.load_dataset("c4", "en",
                                                       keep_in_memory=False, cache_dir=training_args.cache_dir,
                                                       split='train',
                                                       # split='validation',
                                                       streaming=streaming,
                                                       # download_mode="force_redownload",
                                                       ignore_verifications=True,
                                                       )
                title_field, text_field = None, 'text'
            elif dataset_name == 'wiki':
                # https://huggingface.co/datasets/wikipedia
                # size=6,458,670, columns=['id', 'url', 'title', 'text']
                loaded_dataset = datasets.load_dataset("wikipedia", "20220301.en",
                                                       split='train',
                                                       keep_in_memory=False, cache_dir=training_args.cache_dir, streaming=streaming,
                                                       ignore_verifications=True)
                                                       # split=datasets.ReadInstruction('train', from_=0, to=10000, unit='abs'))
                title_field, text_field = 'title', 'text'
            elif dataset_name == 'wiki-dpr':
                # size=21015325, columns=['id', 'title', 'text']
                datapath = '/export/home/data/search/dpr/downloads/data/wikipedia_split/psgs_w100.tsv'
                loaded_dataset = datasets.load_dataset('csv', data_files=datapath, split='train',
                                                       keep_in_memory=False, cache_dir=training_args.cache_dir,
                                                       ignore_verifications=True,
                                                       delimiter="\t" if "tsv" in datapath else ",")
                title_field, text_field = 'title', 'text'
            elif dataset_name == 'pile':
                # https://huggingface.co/datasets/the_pile
                loaded_dataset = datasets.load_dataset("the_pile", cache_dir=training_args.cache_dir, split='train',
                                                       streaming=streaming,
                                                       ignore_verifications=True)
                title_field, text_field = None, 'text'
            elif dataset_name == 'owt2':
                # dataset_size=63.8G
                # train=17,103,059, columns=['title', 'text', 'reddit_scores']
                loaded_dataset = datasets.load_dataset("the_pile_openwebtext2", cache_dir=training_args.cache_dir, split='train',
                                                       streaming=streaming, features=['title', 'text'],
                                                       ignore_verifications=True)
                title_field, text_field = 'title', 'text'
            elif dataset_name == 'pmc':
                # 180.55 GiB
                loaded_dataset = datasets.load_dataset("the_pile", subsets=['pubmed_central'],
                                                       cache_dir=training_args.cache_dir, split='train',
                                                       streaming=streaming,
                                                       ignore_verifications=True)
                title_field, text_field = None, 'text'
            elif dataset_name == 'stackex':
                # https://huggingface.co/datasets/the_pile_stack_exchange
                # train=5,096,117, columns=['domain', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_stack_exchange", cache_dir=training_args.cache_dir,
                                                       split='train', streaming=streaming,
                                                       ignore_verifications=True)
                title_field, text_field = None, 'text'
            elif dataset_name == 'books3':
                # https://huggingface.co/datasets/the_pile_books3
                # train=196,640, columns=['title', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_books3", cache_dir=training_args.cache_dir, split='train',
                                                       streaming=streaming,
                                                       ignore_verifications=True)
                title_field, text_field = None, 'text'
            else:
                assert os.path.isfile(dataset_name), f'{dataset_name} does not exist.'
                print(dataset_name)
                if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
                    loaded_dataset = datasets.load_dataset("json", data_files=dataset_name,
                                                            keep_in_memory=False, cache_dir=training_args.cache_dir,
                                                            streaming=streaming,
                                                            ignore_verifications=True)
                else:
                    loaded_dataset = datasets.load_dataset("text", data_files=dataset_name,
                                                            keep_in_memory=False, cache_dir=training_args.cache_dir,
                                                            streaming=streaming,
                                                            ignore_verifications=True)
                title_field, text_field = None, 'text'
                loaded_dataset = loaded_dataset['train']
            train_datasets.append(loaded_dataset)

        total_train_batch_size = (
                hftraining_args.train_batch_size
                * hftraining_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)
                * moco_args.queue_update_steps
        )
        num_examples = total_train_batch_size * (hftraining_args.max_steps + 100)
        sum_len = sum([len(dset) for dset in train_datasets])
        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            # print(f"  - Fingerprint_name = {fingerprint_name}")
            print(f"  - train_datasets = {training_args.train_file}")
            for d_id, dataset in enumerate(train_datasets):
                print(f"  Num examples in dataset {d_id + 1} = {len(dataset)}")
            print(f"  - prob style = {training_args.train_prob}")
            print(f"  - prob = {str(training_args.train_prob)}")
            print(f"  - Total train_batch_size = {total_train_batch_size}")
            print(f"  \t train_batch_size = {hftraining_args.train_batch_size}")
            print(f"  \t gradient_accumulation_steps = {hftraining_args.gradient_accumulation_steps}")
            print(f"  \t world_size = {(torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)}")
            print(f"  \t queue_update_steps = {moco_args.queue_update_steps}")
            print(f"  - Total optimization steps = {hftraining_args.max_steps}")
            print(f"  \t max_steps = {hftraining_args.max_steps}")
            print(f"  - Total examples in training sets = {sum_len}")
            print(f"  - Total examples for training = {num_examples}")
            print(f"  - Number of epochs for uniform sampling = {(num_examples // sum_len + 1)}")

        if not training_args.train_prob:
            # if train_prob is not set, denotes naive uniform sampling, so simply concatenate them
            if len(train_datasets) > 1:
                print('Concatenate multiple datasets')
                train_dataset = concatenate_datasets(train_datasets)
            else:
                train_dataset = train_datasets[0]
        else:
            probs = [float(p) for p in training_args.train_prob.split(':')] if training_args.train_prob else None
            train_dataset = datasets.interleave_datasets(train_datasets, probabilities=probs,
                                                         seed=hftraining_args.seed, stopping_strategy='all_exhausted')
            '''
            # (@memray slow and prune to OOM) if train_prob is set, sample their indices use our interleave_datasets()
            if training_args.train_prob:
                probs = [float(p) for p in training_args.train_prob.split(':')]
            else:
                if len(train_dataset_names) > 1:
                    print(
                        f'training_args.train_prob is not given, using even sampling probability for {len(train_dataset_names)} datasets: {train_dataset_names}')
                probs = [1 for _ in train_dataset_names]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            fingerprint_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            print("interleave_datasets", f'\n\tdatasets={train_dataset_names}\n\tprobabilities={probs}\n\tnum_step={num_examples}')
            # fingerprint_name = '-'.join(
                # ['data_[' + str(training_args.train_file).replace('\\', '').replace('/', '').replace('pile_', '') + ']',
                #  'prob_' + str(probs),
                #  'num_' + str(num_examples)]).replace(':', ',')
            train_dataset = interleave_datasets(train_datasets,
                                                num_step=num_examples, probabilities=probs,
                                                seed=hftraining_args.seed, new_fingerprint=fingerprint_name[:64])
            '''

        data_prep_config = load_dataprocess_config(training_args, local_rank=hftraining_args.local_rank)
        parse_fn = partial(hfdataset_prepare_features,
                           title_field=title_field,
                           text_field=text_field,
                           num_random_chunk=moco_args.num_random_chunk,
                           **data_prep_config
                           )
        # shuffle will cause OOM if the dataset is huge, say concate 32x of CC
        # print('train_dataset.shuffle')
        # train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)
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

