# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json

def export_cc():
    name2path = {
        'T03B-topic': '/export/home/data/search/upr/cc/T03B_PileCC_topic.jsonl',
        'T03B-title': '/export/home/data/search/upr/cc/T03B_PileCC_title.jsonl',
        'T03B-absum': '/export/home/data/search/upr/cc/T03B_PileCC_absum.jsonl',
        'T03B-exsum': '/export/home/data/search/upr/cc/T03B_PileCC_exsum.jsonl',
        'D2Q-t2q':    '/export/home/data/search/upr/cc/PileCC-doc2query-t2q.jsonl',
        'D2Q-a2t':    '/export/home/data/search/upr/cc/PileCC-doc2query-a2t.jsonl',
        'D2Q-r2t':    '/export/home/data/search/upr/cc/PileCC-doc2query-r2t.jsonl',
    }
    base_reader = open('/export/home/data/search/upr/cc/T03B_PileCC_topic.jsonl', 'r')
    # writer = open('/export/home/data/search/upr/cc/pilecc_uqg.jsonl', 'w')

    name2reader = {}
    for k,p in name2path.items():
        name2reader[k] = open(p, 'r')

    for lid, l in enumerate(base_reader):
        base_dict = json.loads(l)
        docid = base_dict['id'][base_dict['id'].rfind('-') + 1:]
        del base_dict['output-prompt0']
        outputs = {}
        for k, r in name2reader.items():
            uqg_line = r.readline()
            uqg_dict = json.loads(uqg_line)
            _docid = uqg_dict['id'][uqg_dict['id'].rfind('-') + 1:]
            assert docid == _docid
            outputs[k] = uqg_dict['output-prompt0']
        base_dict['outputs'] = outputs
        # writer.write(json.dumps(base_dict)+'\n')
        if lid % 1 == 0:
            print(lid)
            print(json.dumps(base_dict, indent=4))
        if lid % 100 == 0:
            print(lid)
            print(json.dumps(base_dict, indent=4))
            # break
    print('Export done, #line=', lid)
    # writer.close()


def export_wiki():
    name2path = {
        'T03B-topic': '/export/home/data/search/upr/wikipsg/T03B_wikipsg_topic_shard100k.jsonl',
        'T03B-title': '/export/home/data/search/upr/wikipsg/T03B_wikipsg_title_shard100k.jsonl',
        'T03B-absum': '/export/home/data/search/upr/wikipsg/T03B_wikipsg_absum_shard100k.jsonl',
        'T03B-exsum': '/export/home/data/search/upr/wikipsg/T03B_wikipsg_exsum_shard100k.jsonl',
        'D2Q-t2q':    '/export/home/data/search/upr/wikipsg/doc2query-t2q-wikipsg-shard100k.jsonl',
        'D2Q-a2t':    '/export/home/data/search/upr/wikipsg/doc2query-a2t-wikipsg-shard100k.jsonl',
        'D2Q-r2t':    '/export/home/data/search/upr/wikipsg/doc2query-r2t-wikipsg-shard100k.jsonl',
    }
    base_reader = open('/export/home/data/search/upr/wikipsg/T03B_wikipsg_title_shard100k.jsonl', 'r')
    writer = open('/export/home/data/search/upr/wikipsg/wiki_uqg.jsonl', 'w')

    name2reader = {}
    for k,p in name2path.items():
        name2reader[k] = open(p, 'r')
    for lid, l in enumerate(base_reader):
        base_dict = json.loads(l)
        docid = base_dict['id'][base_dict['id'].rfind('-') + 1:]
        del base_dict['output-prompt0']
        outputs = {}
        for k, r in name2reader.items():
            uqg_line = r.readline()
            uqg_dict = json.loads(uqg_line)
            _docid = uqg_dict['id'][uqg_dict['id'].rfind('-') + 1:]
            assert docid == _docid
            outputs[k] = uqg_dict['output-prompt0']
        base_dict['outputs'] = outputs
        writer.write(json.dumps(base_dict)+'\n')
        # if lid % 10000 == 0:
        #     print(lid)
        #     print(json.dumps(base_dict, indent=4))
        if lid % 100000 == 0:
            print(lid)
            print(json.dumps(base_dict, indent=4))
            # break
    print('Export done, #line=', lid)
    writer.close()


if __name__ == '__main__':
    # export_wiki()
    export_cc()
