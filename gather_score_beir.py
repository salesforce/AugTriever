# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import os.path
import pandas as pd

'''
4 datasets are not directly available by BEIR: 'bioasq', 'signal1m', 'robust04', 'trec-news'
2 datasets are not in leaderboards by Mar 19, 2022: 'cqadupstack', 'quora'
'''

def main():
    exp_base_dir = '/export/home/exp/search/unsup_dr/augtriever-release/'
    exp_names = [
        'cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'
    ]

    # exp_base_dir = '/export/home/exp/search/unsup_dr/baselines/'
    # exp_names = [
    #     'e5-base-unsupervised.cosine',
    #     'all-mpnet-base-v2.cosine'
    # ]

    if False:
        exp_base_dir = '/export/home/exp/search/unsup_dr/cc_v2/'
        exp_names = [
            # 'cc.RC50-Topic50.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5'
            # 'cc.RC50-Topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs4096.lr5e5'

            # T2Q
            # 'cc.RC50-T2Q50.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5'
            # 'cc.RC50-T2Q50_minlen0.05.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'

            # T0gen
            # 'cc.RC20+T0gen80.seed477.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5',
            # 'cc.RC20+T0gen80.moco-2e14.contriever256.bert-base-uncased.avg.dot.qd128.step100k.bs8192.lr5e5'
            # 'cc.RC20+T0gen80_minlen0.05.moco-2e16.contriever256.bert-base-uncased.avg.dot.len224qd128.step100k.bs8192.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e16.contriever256.bert-base-uncased.avg.dot.len256qd128.step200k.bs4096.lr5e5'
            # 'cc.RC20+T0gen80.noconcat.moco-2e16.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs8192.lr5e5'
            # 'cc.RC20+T0gen80.interleave.moco-2e16.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs8192.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e16.momentum995.contriever256.bert-base-uncased.avg.dot.len256qd128.step200k.bs4096.lr5e5'

            # 'cc.RC20+T0gen80.moco-2e15.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs4096.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e16.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs4096.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e17.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs4096.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e17.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs2048.lr5e5'


            # large
            # 'cc.RC20+T0gen80.moco-2e14.contriever256-special50.bert-large-uncased.avg.dot.q128d128.step100k.bs2048.lr1e5'
            # 'cc.RC20+T0gen80.moco-2e14.contriever256.bert-large-uncased.avg.dot.q128d128.step100k.bs2048.lr3e5'
            # 'cc.RC20+T0gen80.moco-2e14.contriever256-special50.bert-large-uncased.cls.dot.q128d128.step100k.bs2048.lr1e5'
            # 'cc.RC20+T0gen80.moco-2e14.contriever256.bert-large-uncased.avg.dot.q128d128.step200k.warmup20k.bs2048.lr5e5'
            # 'cc.RC20+T0gen80.moco-2e16.contriever256.bert-large-uncased.avg.dot.q128d128.step300k.warmup30k.bs2048.lr5e5'
            # 'cc.T0gen.inbatch.contriever256.bert-large-uncased.cls.mlp+none.dot.q128d128.step100k.bs1024.lr5e5'

            # Data mixing
            # 'cc.T0gen.seed477.moco-2e14.contriever256.bert-base-uncased.avg.dot.len192q96d128.step200k.bs4096.lr5e5'
            # 'medi.inbatch+neg.inbatch.bert-base-uncased.avg.dot.q128d128.step100k.bs2048.lr5e5'
            # 'wiki+cc.RC20+T0gen80.moco-2e14.contriever256.bert-base-uncased.avg.dot.len256qd128.step200k.bs4096.lr5e5'
            # 'pile8-uniform.moco-2e14.contriever256.title50.bert-base-uncased.avg.dot.qd128.step100k.bs4096.lr5e5'

            # Other backbones
            # 'cc.RC20+T0gen80.moco-2e14.contriever256.roberta-large.avg.dot.q128d128.step100k.warmup10k.bs2048.lr1e5'
            # 'cc.RC20+T0gen80.noconcat.moco-2e16.contriever256.roberta-base.avg.dot.len256qd128.step100k.bs8192.lr5e5'
            # 'cc.RC20+T0gen80.inbatch.contriever256.deberta-v3-base.cls.dot.len256qd128.step100k.bs2048.lr1e5',
            # 'cc.RC20+T0gen80.inbatch.contriever256.roberta-base.cls+mlp.dot.len256qd128.step100k.bs1024.lr1e5',
            # 'cc.RC20+T0gen80.inbatch.contriever256.roberta-base.cls.dot.len256qd128.step100k.bs4096.lr1e5',
            # 'wiki20+cc80.RC20+T0gen80.moco-2e14.contriever256.bert-base-uncased.avg.dot.len256qd128.step100k.bs2048.lr5e5',
            # 'cc.RC20+T0gen80.inbatch.contriever256.deberta-v3-base.cls.dot.len256qd128.step100k.bs2048.lr5e5'
            # 'cc.RC20+T0gen80.inbatch.contriever256.roberta-base.cls.dot.len256qd128.step100k.bs4096.lr5e5'
            ]

    if False:
    # if True:
        exp_base_dir = '/export/home/exp/search/unsup_dr/wikipsg_v1/'
        exp_names = [
            # 'wikipsg.seed477.inbatch.contriever256.bert-base-uncased.avg.dot.qd256.step100k.bs1024.lr5e5',
            # 'wikipsg.seed477.moco-2e14.contriever256.bert-base-uncased.avg.dot.qd256.step100k.bs1024.lr5e5',

            # 'wikipsg.seed477.inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'wikipsg.seed477.inbatch.contriever-256-Qtitle50.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'wikipsg.seed477.inbatch.contriever-256-Qtitle1.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'wikipsg.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'wikipsg.seed477.moco-2e14.contriever-256-Qtitle50.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'wikipsg.seed477.moco-2e14.contriever-256-Qtitle1.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',

            # 'wiki.extphrase3.seed477.inbatch.bert-base-uncased.avg.dot.d128d256.step100k.bs1024.lr5e5',
            # 'wiki.extphrase3-50.seed477.inbatch.bert-base-uncased.avg.dot.d128d256.step100k.bs1024.lr5e5',
            # 'wiki.extphrase3.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.extphrase3-50.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'wiki.extphrase3.seed477.inbatch.bert-base-uncased.avg.dot.d128d256.step100k.bs1024.lr5e5',
            # 'wiki.extphrase3.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'wiki_allphrase1.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_allphrase1.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_allphrase3.seed477.inbatch.bert-base-uncased.avg.dot.d128d256.step100k.bs1024.lr5e5',
            # 'wiki_allphrase3.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_allphrase5.seed77.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_allphrase5.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'wiki.ExtQ-selfdot-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-selfdot-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ-bm25-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-bm25-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ-plm-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-plm-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-plm-chunk16.seed477.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'wiki.ExtQ-selfdot-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-selfdot-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ-bm25-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-bm25-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ-plm-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.ExtQ50-plm-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'paq.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'paq.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5'

            # 'wiki.T03b_topic.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_topic.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_topic50.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_topic50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_topic50.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_topic50.seed477.moco-2e14.wikipsg256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_topic.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_topic.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

            # 'wiki.T03b_title.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_title.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_title.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_title50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_title.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_title50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

            # 'wiki.T03b_absum.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_absum.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_absum.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_absum50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_absum.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_absum50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

            # 'wiki.T03b_exsum.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.T03b_exsum.seed477.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_exsum50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_exsum50.seed77.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_exsum.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki_T03b_exsum50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

            # 'wiki.doc2query-t2q.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_doc2query50_t2q.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5'
            # 'wiki.doc2query-t2q.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki_doc2query50_t2q.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'wiki.doc2query50-t2q.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'wiki.doc2query50-t2q.seed477.moco-2e14.wikipsg256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'paq-cropdoc.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq-cropdoc.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq50.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq50.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq-fulldoc.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq-fulldoc.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq50-fulldoc.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq50-fulldoc.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'paq.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
            # 'paq.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

            # 'wikipsg.seed477.moco-inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'wikipsg.seed477.moco-inbatch-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'wikipsg.seed477.moco-2e17.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'cc+wikipsg.equal.inbatch.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5'
            ]

    # if True:
    if False:
        exp_base_dir = '/export/home/exp/search/unsup_dr/cc_v1/'
        exp_names = [
            # 'cc.inbatch.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.inbatch.contriever256-Qtitle50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'cc.inbatch.contriever256-Qtitle.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.moco-2e14.wikipsg256-Qtitle05.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5',
            # 'cc.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'rerun.cc.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.moco-2e14.contriever-256-Qtitle05-Aug.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.moco-2e14.contriever256-Qtitle05-AugDel02.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',

            # 'cc.ExtQ-bm25-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ50-bm25-chunk16.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ-bm25-chunk16.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ50-bm25-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ-plm-chunk16.seed47-rerun.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ50-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ50-plm-chunk16.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ-plm-chunk16.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.ExtQ50-plm-chunk16.seed47-rerun.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'cc+wikipsg.equal.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs2048.lr5e5',
            # 'cc.inbatch.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'cc.inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'cc.inbatch-indep.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
            # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd224.step200k.bs2048.lr5e5',
            # 'cc.moco-2e17.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs2048.lr5e5',

            # 'cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic10.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic90.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic25.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'cc.T03b_topic50.inbatch.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic75.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'cc.T03b_topic.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5',

            # 'cc.T03b_title.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_title50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'cc.T03b_title50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5'

            # 'cc.T03b_absum.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_absum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            'cc.T03b_absum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5',
            # 'cc.T03b_absum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5',

            # 'cc.T03b_exsum.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_exsum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.T03b_exsum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5',
            # 'cc.T03b_exsum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5',

            # 'cc.d2q-t2q.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.d2q-t2q50.inbatch.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.doc2query50-t2q.inbatch.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.doc2query50-t2q.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.d2q-t2q50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc.d2q-t2q50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'cc.doc2query50-t2q.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',

            # doc length ablation
            # 'cc-len64.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q64d384.step100k.bs512.lr5e5',
            # 'cc-len128.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q64d384.step100k.bs512.lr5e5',
            # 'cc-len384.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q64d384.step100k.bs512.lr5e5',

            # hybrid
            # 'cc-hybrid.RC50+title10+T0gen40.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc-hybrid.RC20+title16+T0gen64.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc-hybrid.Qext0+title20+topic40+exsum40.seed477.inbatch.contriever256-special-titles_special.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'cc-hybrid.Qext0+title20+topic40+exsum40.seed477.moco-2e14.contriever256-special-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'cc-hybrid.Qext50+title20+topic40+exsum40.seed477.inbatch.contriever256-special-titles_special.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
            # 'cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc-hybrid.Qext0+title20+topic40+exsum40.seed477.moco-2e14.contriever256-special-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc-hybrid.RC20+Qext10+title16+T0gen64.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'cc-hybrid.RC50+Qext10+title10+T0gen40.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'cc-hybrid.Qext50+title20+topic40+exsum40.seed477.moco-2e14.contriever256-special-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            'cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'
            # 'cc-hybrid.RC20+Qext10+title16+T0gen64.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step200k.bs1024.lr5e5',
            # 'cc-hybrid.RC20+Qext10+title16+T0gen48+d2q16.seed477.moco-2e14.contriever256-special80-titles_special.bert-base-uncased.avg.dot.q128d256.step200k.bs1024.lr5e5',

            # pile
            # 'pile6-uniform.moco-2e14.contriever256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'pile10-uniform.moco-2e14.contriever256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
        ]

    # if True:
    if False:
        exp_base_dir = '/export/home/exp/search/unsup_dr/wikipsg_v1-FT/'
        exp_names = [
            # baseline
            # 'FT-inbatch-random-neg1023+1024.spar-wiki-context.avg.dot.qd192.step20k.bs1024.lr1e5',
            # 'FT-inbatch-random-neg1023+1024.spar-wiki-query.avg.dot.qd192.step20k.bs1024.lr1e5',
            # 'FT-inbatch-random-neg1023+1024.facebook-contriever.rerun.avg.dot.qd192.step20k.bs1024.lr1e5'
            # 'FT-inbatch-random-neg1023+1024.tau-spider.avg.dot.qd192.step20k.bs1024.lr1e5',
            # 'FT-inbatch-random-neg1023+1024.mm.inbatch.facebook-contriever.avg.dot.qd192.step20k.bs1024.lr1e5',
            # 'FT-inbatch-random-neg1023+1024.spar-wiki-context.cls.dot.qd192.step20k.bs1024.lr1e5', # trash
            # 'FT-inbatch-random-neg1023+1024.spar-wiki-context.avg.actually-cls.dot.qd192.step20k.bs1024.lr1e5'

            # RC
            # 'FT-inbatch-random-neg1023+1024.cc.inbatch.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.cc.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'

            # title
            # 'FT-inbatch-random-neg1023+1024.cc.inbatch.contriever256-Qtitle.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.wiki.T03b_topic50.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # Qext-PLM
            # 'FT-inbatch-random-neg1023+1024.lr5e5.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.ExtQ50-plm-chunk16.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.lr5e6.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.lr2e5.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # topic
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',

            # title
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_title.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_title50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'

            # absum

            # exsum
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_exsum.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.cc.T03b_exsum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            
            # D2Q
            # 'FT-inbatch-random-neg1023+1024.cc.d2q-t2q.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-random-neg1023+1024.cc.d2q-t2q50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5',

            # hybrid
            # 'FT-inbatch-random-neg1023+1024.cc-hybrid.RC20+Qext10+title16+T0gen64.seed477.moco-2e14.contriever256-special50-titles_special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'FT-inbatch-random-neg1023+1024.cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            'FT-inbatch-random-neg1023+1024.cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'

            # deprecated
            # 'FT-inbatch-random-neg1023+1024.wikipsg.seed477.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'FT-inbatch-wikipsg.seed477.moco-2e14.contriever256-Qtitle50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            ]
    # if True:
    if False:
        exp_base_dir = '/export/home/exp/search/unsup_dr/cc_v1-DA/'
        exp_names = [
            # 'DA-plm12-inbatch-msmarco.lr1e5.step5k.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-plm12-inbatch-msmarco.lr1e5.step1k.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-plm12-inbatch-msmarco.lr1e5.step2k.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-plm12-inbatch-msmarco.lr5e5.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-plm12-inbatch-msmarco.lr1e5.step1k.warmup100.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-plm12-inbatch-msmarco.lr1e5.step2k.warmup100.cc.ExtQ-plm-chunk16.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_exsum-inbatch-msmarco.lr1e5.step2k.cc.T03b_exsum.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_exsum-moco-msmarco.lr1e5.step2k.cc.T03b_exsum50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-msmarco.lr1e5.step2k.cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5'
            # 'DA-T03b_topic-moco-msmarco.lr1e5.step2k.cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
            # 'DA-T03b_topic-inbatch-msmarco.lr1e5.step1k.cc.T03b_topic50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5'
            # 'DA-T03b_topic-inbatch-msmarco.lr1e5.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-msmarco.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-cqadupstack.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-quora.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-nfcorpus.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-fiqa.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-webis_touche2020.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-trec_covid.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-scidocs.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-arguana.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-wiki.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-scifact.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'

            # 'DA-T03b_topic-inbatch-arguana.lr5e6.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-webis_touche2020.lr5e6.step1k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-trec_covid.lr5e6.step1k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

            # 'DA-T03b_topic-inbatch-arguana.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-trec_covid.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-fiqa.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-webis_touche2020.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-scidocs.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-sci.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-scifact.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-nfcorpus.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
            # 'DA-T03b_topic-inbatch-webis_touche2020.lr5e6.step2k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
            # 'DA-T03b_topic-inbatch-cqadupstack_quora.lr1e5.step5k.cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'

        ]

    beir_datasets = [
        'msmarco',
        'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04',
        'quora', 'cqadupstack']

    beir_metric_cats = ['ndcg', 'recall', 'map', 'mrr', 'precision', 'recall_cap', 'hole']
    beir_metrics = ['ndcg', 'recall', 'map', 'mrr', 'p', 'r_cap', 'hole']
    exp2scores = {}

    for exp_name in exp_names:
        print('=-' * 20)
        print(exp_name.upper())
        print('=-' * 20)
        header_row = None
        data2scores = {}

        for dataset in beir_datasets:
            if not header_row: _header_row = ['']
            score_dict = {}
            for metric_prefix in beir_metrics:
                for k in [1, 3, 5, 10, 100, 1000]:
                    score_dict[f'{metric_prefix}@{k}'] = 0.0
            data2scores[dataset] = score_dict

            # score_json_path = os.path.join(exp_base_dir, exp_name, f'{dataset}.json')
            score_json_path = os.path.join(exp_base_dir, exp_name, 'beir_output', f'{dataset}.json')
            if not os.path.exists(score_json_path):
                print(f'{dataset} not found at: {score_json_path}')
            else:
                print(dataset.upper())
                with open(score_json_path, 'r') as jfile:
                    result_data = json.load(jfile)
                # print(result_data)

                for metric_prefix in beir_metric_cats:
                    for metric, score in result_data['scores'][metric_prefix].items():
                        if metric.lower() not in score_dict: continue
                        score_dict[metric.lower()] = score

        exp2scores[exp_name] = pd.DataFrame.from_dict(data2scores)

    for exp_name, score_pd in exp2scores.items():
        print('*' * 20)
        print(exp_name)
        print(score_pd.to_csv())


if __name__ == '__main__':
    main()
