# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json

def export_shards2jsonl():
    input_path = '/export/home/data/pretrain/medi/medi-data.json'
    output_path = '/export/home/data/pretrain/medi/medi-data.jsonl'

    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        print("Reading file %s" % input_path)
        data = json.load(f)
        results.extend(data)
        print("Aggregated data size: {}".format(len(results)))

    with open(output_path, 'w') as writer:
        for i, ex in enumerate(results):
            if i % 10000 == 0:
                print(i)
            writer.write(json.dumps(ex) + '\n')

    print('Done')


if __name__ == '__main__':
    export_shards2jsonl()
