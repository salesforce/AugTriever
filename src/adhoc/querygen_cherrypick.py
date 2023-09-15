# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import os.path
from collections import defaultdict

dir_path = '/export/home/data/search/upr/wikipsg'
file_list = [
    'T03B_wikipsg_title_shard100k.jsonl',
    'T03B_wikipsg_topic_shard100k.jsonl',
    'T03B_wikipsg_exsum_shard100k.jsonl',
    'T03B_wikipsg_absum_shard100k.jsonl',
    'doc2query-t2q-wikipsg-shard100k.jsonl',
    'doc2query-a2t-wikipsg-shard100k.jsonl',
    'doc2query-r2t-wikipsg-shard100k.jsonl',
    't5xl-insummary-wikipsg-shard100k.jsonl',
]

# dir_path = '/export/home/data/search/upr/cc'
# file_list = [
#     'T03B_PileCC_title.json',
#     'T03B_PileCC_topic.json',
#     'T03B_PileCC_exsum.json',
#     'T03B_PileCC_absum.json',
#     'PileCC-doc2query-t2q.jsonl',
# ]

id2output = defaultdict(dict)
for fname in file_list:
    path = os.path.join(dir_path, fname)
    print(fname)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000: break
            d = json.loads(line)
            # print(d)
            id2output[d['text']][fname] = d['output-prompt0']
            id2output[d['text']]['title'] = d['title']
            id2output[d['text']]['ext_phrases'] = d['anchor_phrases'] + d['font_phrases']

for cnt, (text, outputs) in enumerate(id2output.items()):
    if cnt < 50: continue
    print(cnt)
    print('=' * 50, cnt, '=' * 50)
    print(text)
    print('-' * 50)
    for k, v in outputs.items():
        print(k, ':', v)
    print('=' * 100)