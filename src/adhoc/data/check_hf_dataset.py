# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter

import datasets

# train_dataset = datasets.load_dataset("the_pile", cache_dir='/export/home/data/pretrain/.cache', split='train[:1%]', streaming=False)
train_dataset = datasets.load_dataset("the_pile", cache_dir='/export/home/data/pretrain/.cache', split='train', streaming=True)
# train_dataset = datasets.load_dataset("c4", 'en', cache_dir='/export/home/data/pretrain/.cache', split='train', streaming=True)
# train_dataset = datasets.load_dataset("c4", 'en', cache_dir='/export/home/data/pretrain/.cache', split='train', streaming=True)

# pile_dataset = 'Pile-CC'
# train_dataset = datasets.load_dataset("json", data_files=f'/export/home/data/pretrain/pile/{pile_dataset}.json', keep_in_memory=False, cache_dir='/export/home/data/pretrain/.cache', streaming=True)
# train_dataset = train_dataset['train']

train_dataset = train_dataset.shuffle(seed=42)
subset_count = Counter()
length_count = Counter()

for i, ex in enumerate(train_dataset):
    if i % 10000 == 0:
        print(i)
    if i >= 1000000:
        break
    # subset_count[ex['meta']['pile_set_name']] += 1
    len_text = len(ex['text'].split())
    length_count[len_text // 100 * 100] += 1

    title = ''
    if 'meta' in ex:
        if ex['meta']['pile_set_name'] == 'Github':
            title = ''
        elif ex['meta']['pile_set_name'] in ['Wikipedia (en)', 'Pile-CC', 'OpenWebText2']:
            lines = [l for l in ex['text'].split('\n') if len(l.strip()) > 0]
            if len(lines) > 0: title = lines[0]
        else:
            lines = [l for l in ex['text'].split('\n') if len(l.strip().split()) > 3]
            if len(lines) > 0: title = lines[0]

    if 'meta' in ex: print('pile_set_name\t=\t', ex['meta']['pile_set_name'])
    print('title\t=\t', title)
    print(ex['text'])
    # print(ex)
    # print(len_text)
    print('=' * 50)
    print()

print('subset_count')
print(subset_count)
for k, v in sorted(subset_count.items(), key=lambda k:k[1], reverse=True):
    print(f'\t{k} - {v}')

print('length_count')
print(length_count)
for k, v in sorted(length_count.items(), key=lambda k:k[0]):
    print(f'\t{k} - {v}')
