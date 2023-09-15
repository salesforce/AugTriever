# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
Process UQG output, assign new doc ID
'''
import json
import os

input_dir = '/export/home/data/search/upr/beir/topic'
output_dir = '/export/home/data/search/upr/beir/topic/cleaned_jsonl'

for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    if not filename.endswith('.jsonl') or os.path.isdir(filepath): continue
    print(filename)
    count = 0
    new_lines = []
    with open(filepath, 'r') as f:
        for l in f.readlines():
            count += 1
            ex = json.loads(l)
            new_ex = {'id': ex['id'], '_id': ex['_id'], 'title': ex['title'], 'text': ex['text'], 'output-prompt0': ex['output-prompt0']}
            new_lines.append(new_ex)
        print(filepath, len(new_lines))

    outpath = os.path.join(output_dir, filename)
    with open(outpath, 'w') as f:
        for ex in new_lines:
            f.write(json.dumps(ex) + '\n')

print('Done')
