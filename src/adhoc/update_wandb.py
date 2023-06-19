# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


# each triple contains (wandb_runid, max_step, exp_path)
import os.path

from transformers.integrations import rewrite_logs

from utils import eval_utils

'''
doesn't work, wandb currently only supports adding new metric values, but cannot remove them
change x-axis to `steps` will show all values, `global_step` will only show the oldest one
'''
prev_exps=[
    ('1bmzlxj8', 200000, 'wikipsg.seed477.inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5'),
    ('370bk36z', 200000, 'wikipsg.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5'),
]

runid, max_steps, exp_name = prev_exps[0]
exp_base_dir = '/export/home/exp/search/unsup_dr/wikipsg_v1/'
exp_dir = os.path.join(exp_base_dir, exp_name)

# reload beir
results = {}
beir_output_dir = os.path.join(exp_base_dir, exp_name, 'beir_output')
results_beir, _ = eval_utils.load_prev_beir_results(beir_output_dir)
results.update(results_beir)

# reload qa
# reload senteval
results = rewrite_logs(results)
results['train/global_step'] = max_steps

wandb_dir = os.path.join(exp_base_dir, exp_name, 'wandb', 'latest-run')
os.environ["WANDB_DIR"] = exp_dir
import wandb
api = wandb.Api()
run = api.run(path="memray/unsup_retrieval_wikipsg/1bmzlxj8")
run.dir = wandb_dir
for k,v in results.items():
    print(k, v, run.summary[k])
    run.summary[k] = v
run.summary.update()

# print(run.history(keys=["eval/beir-msmarco_ndcg@10"]))
