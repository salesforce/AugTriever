# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
deduplicate documents, Pile contains massive duplicates
'''
import json
import os


def clean_wiki():
    input_wiki_file = '/export/home/data/pretrain/pile/Wikipedia.json'
    output_wiki_file = '/export/home/data/pretrain/pile/Wikipedia_dedup.json'

    url2row = {}
    num_rows = 0
    with open(input_wiki_file, 'r') as input:
        for r in input:
            ex = json.loads(r)
            title = [l.strip() for l in ex['text'].split('\n') if len(l.strip()) > 0][0]
            url2row[title] = r
            num_rows += 1

    print('#row=', num_rows)
    print('#dedup_row=', len(url2row))

    with open(output_wiki_file, 'w') as output:
        for r in url2row.values():
            output.write(r + '\n')


def clean_pile():
    input_pile_folder = '/export/home/data/pretrain/pile_new_dedup'
    output_pile_folder = '/export/home/data/pretrain/pile_new'
    os.makedirs(output_pile_folder, exist_ok=True)

    for filename in os.listdir(input_pile_folder):
        if not filename.endswith('.json'): continue
        print('*' * 20)
        print(filename)
        input_file = os.path.join(input_pile_folder, filename)
        output_file = os.path.join(output_pile_folder, filename)
        url2row = {}
        num_rows = 0
        with open(input_file, 'r') as input:
            for r in input:
                if not r.strip(): continue # empty lines
                ex = json.loads(r)
                lines = [l.strip() for l in ex['text'].split('\n') if len(l.strip()) > 0]
                key = '!'.join(lines[:3])[:200]
                url2row[key] = r
                num_rows += 1

        print('#row=', num_rows)
        print('#unique_row=', len(url2row))

        with open(output_file, 'w') as output:
            for r in url2row.values():
                output.write(r.strip() + '\n')


if __name__ == '__main__':
    # clean_wiki()
    clean_pile()
