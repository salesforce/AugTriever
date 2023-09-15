# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import os
from scipy import stats
from src.wiki.wiki_utils import extract_phrases

input_dir = '/export/share/ruimeng/data/wiki/phrase/'
output_path = '/export/home/data/search/wiki/wiki_phrase.jsonl'

shard_path = []
os.chdir(input_dir)
lines = []
doc_lens = []

for subdir in sorted(os.listdir(input_dir)):
    for file in sorted(os.listdir(subdir)):
        if file.endswith(".json"):
            file_path = os.path.join(input_dir, subdir, file)
            print(file_path)
            with open(file_path, 'r', encoding='utf-8') as shard:
                for l in shard:
                    ex = json.loads(l)
                    ex['wiki_text'] = ex['text']
                    plain_text, font_phrases, anchor_phrases = extract_phrases(ex['text'])
                    ex['text'] = plain_text
                    ex['font_phrases'] = font_phrases
                    ex['anchor_phrases'] = anchor_phrases
                    lines.append(json.dumps(ex))
                    doc_lens.append(len(plain_text.split()))
            print('#lines=', len(lines))


print('Dumping to ', output_path)
with open(output_path, 'w') as writer:
    for i, l in enumerate(lines):
        if i % 1000000 == 0:
            print(i)
        writer.write(l + '\n')

print(stats.describe(doc_lens))
