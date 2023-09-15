# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import os.path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig

from src.model.inbatch import InBatch
from src.model.moco import MoCo
from src.utils import eval_utils
from src.utils.training_utils import reload_model_from_ckpt


def report_senteval(json_path):
    data2scores = {}
    try:
        with open(json_path, 'r') as json_file:
            senteval_scores = json.load(json_file)
        for k,v in senteval_scores.items():
            if k.startswith('eval_senteval-'):
                dataset = k[14:]
                if dataset in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'avg_transfer']:
                    data2scores[dataset] = v
                else:
                    data2scores[dataset] = v * 100.0
        return data2scores
    except Exception as e:
        print('Error loading', json_path)
        return {}


def run_senteval(exp_dir):
    if not os.path.exists(exp_dir):
        print('NOT FOUND', exp_dir)
        return
    _, _, moco_args = torch.load(os.path.join(exp_dir, "model_data_training_args.bin"))
    hf_config = AutoConfig.from_pretrained(moco_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(moco_args.model_name_or_path, cache_dir='/export/home/data/pretrain/.cache', use_fast=True)
    if moco_args.arch_type == 'moco':
        model = MoCo(moco_args, hf_config)
    elif moco_args.arch_type == 'inbatch':
        model = InBatch(moco_args, hf_config)
    else:
        raise NotImplementedError(f'Unknown arch type {hf_config.arch_type}')
    reload_model_from_ckpt(model, exp_dir)
    model = model.cuda()
    results_senteval = eval_utils.evaluate_senteval(model, tokenizer,
                                                    output_dir=exp_dir + '/senteval_output',
                                                    eval_senteval_sts_all=True,
                                                    eval_senteval_transfer=True,
                                                    )
    print('*' * 30)
    print(exp_dir)
    print(results_senteval)
    print('*' * 30)


def main():
    exp_base_dir = '/export/home/exp/search/unsup_dr/augtriever-release/'
    exp_names = [
        'cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'
    ]

    score_dicts = []
    for exp_name in exp_names:
        print('=-' * 20)
        print(exp_name)
        print('=-' * 20)
        # run_senteval(os.path.join(exp_base_dir, exp_name))
        senteval_path = os.path.join(exp_base_dir, exp_name, 'senteval_output/senteval.json')
        data2scores = report_senteval(senteval_path)
        data2scores['exp'] = exp_name
        score_dicts.append(data2scores)
    cols = ['exp',
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
            'STSBenchmark', 'SICKRelatedness',
            'avg_sts_7', 'avg_sts',
            'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC',
            'avg_transfer']
    df = pd.DataFrame.from_records(score_dicts, columns=cols, index='exp')

    print(df.to_csv())


if __name__ == '__main__':
    main()
