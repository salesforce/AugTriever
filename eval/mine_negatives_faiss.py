# Code adapted from Contriever (https://github.com/facebookresearch/contriever/).
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import argparse
import torch
import logging
import json
import os
os.environ['OMP_NUM_THREADS'] = str(32)

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from src import dist_utils
import beir.util

from src.beir.retrieval.evaluation import EvaluateRetrieval
from src.beireval import slurm, beir_utils
from src.moco import MoCo
from src.utils import init_logger

from src.beir.datasets.data_loader import GenericDataLoader
from src.beir.retrieval.search.dense import DenseRetrievalExactSearch, FlatIPFaissSearch

logger = logging.getLogger(__name__)

def setup(args):
    slurm.init_distributed_mode(args)
    slurm.init_signal_handler()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = init_logger(args)
    logger.info(f"Loading model from {args.model_name_or_path}")

    moco_args = torch.load(os.path.join(args.model_name_or_path, "model_data_training_args.bin"))[-1]
    hf_config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = MoCo(moco_args, hf_config)
    state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location="cpu")
    # back compatibility: for loading old ckpts
    if 'queue_k' not in state_dict:
        items = list(state_dict.items())
        for k, v in items:
            if k.startswith('encoder_q'):
                _k = k.replace('encoder_q', 'encoder_q.model')
                state_dict[_k] = v
                del state_dict[k]
            elif k.startswith('encoder_k'):
                _k = k.replace('encoder_k', 'encoder_k.model')
                state_dict[_k] = v
                del state_dict[k]
        state_dict['queue_k'] = state_dict['queue']
        del state_dict['queue']
    model.load_state_dict(state_dict, strict=True)
    if args.use_gpu:
        model = model.cuda()
        model = model.half()
    return tokenizer, model


def mine_msmarco(args, tokenizer, model):
    args.dataset = 'msmarco'
    # args.dataset = 'trec-covid'
    # args.dataset = 'nfcorpus'
    # args.dataset = 'scifact'
    logger.info(f"Start indexing with dataset={args.dataset}")
    split = 'train' if args.dataset == 'msmarco' else 'test'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = beir.util.download_and_unzip(url, args.beir_data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    metric = 'cos_sim' if model.sim_metric == 'cosine' else 'dot'
    if args.use_faiss:
        dmodel = FlatIPFaissSearch(
            beir_utils.DenseEncoderModel(
                query_encoder=model,
                doc_encoder=model,
                tokenizer=tokenizer,
                maxlength=512,
                add_special_tokens=True,
                norm_query=model.norm_query,
                norm_doc=model.norm_doc,
            ),
            batch_size=args.per_gpu_batch_size,
            query_batch_size=8,  # faiss bug? batch size must be small if use_gpu=True. large batch size leads to zero scores
            use_gpu=False,
            add_qd_prompt=args.add_qd_prompt,
            corpus_chunk_size=16384
        )
        # dmodel.index(corpus, metric)
        # dmodel.save(args.output_dir, args.dataset, split)
        dmodel.load(args.output_dir, args.dataset, split)
    else:
        dmodel = DenseRetrievalExactSearch(
            beir_utils.DenseEncoderModel(
                query_encoder=model,
                doc_encoder=model,
                tokenizer=tokenizer,
                maxlength=512,
                add_special_tokens=True,
                norm_query=model.norm_query,
                norm_doc=model.norm_doc,
            ),
            return_cpu=True,
            batch_size=args.per_gpu_batch_size,
            query_batch_size=4096,
            add_qd_prompt=args.add_qd_prompt,
            corpus_chunk_size=8192
        )
    retriever = EvaluateRetrieval(dmodel, score_function=metric, k_values=[10])
    predicts = retriever.retrieve(corpus, queries)
    # load again to remove prompts
    # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    if dist_utils.is_main():
        ndcg, _map, recall, precision = retriever.evaluate(qrels, predicts, k_values=[5, 10, 100])
        print(f'Dumping negatives to {args.output_dir}/{args.dataset}.jsonl')
        progress_bar = tqdm(range(len(qrels)), desc=f"Creating DPR formatted {args.dataset} file")
        with open(f'{args.output_dir}/{args.dataset}.jsonl', 'w') as fp:
            for cnt, (query_id, pos_doc2score) in enumerate(qrels.items()):
                # query
                query = queries[query_id]
                # positive doc
                pos_doc_id, pos_score = list(pos_doc2score.items())[0]
                pos_ctx = corpus[pos_doc_id]
                pos_ctx['id'] = pos_doc_id
                pos_ctx['score'] = pos_score
                # negative docs
                neg_ctxs = []
                pred_d2scores = sorted(predicts[query_id].items(), key=lambda k: k[1], reverse=True)
                for neg_id, score in pred_d2scores[:args.num_negatives]:
                    neg_ctx = corpus[neg_id]
                    neg_ctx['id'] = neg_id
                    neg_ctx['score'] = score
                    neg_ctxs.append(neg_ctx)

                json.dump({"id": query_id,
                           "question": query,
                           "positive_ctxs": [pos_ctx],
                           "hard_negative_ctxs": neg_ctxs}, fp)
                fp.write("\n")
                progress_bar.update(1)


def main(args):
    tokenizer, model = setup(args)
    mine_msmarco(args, tokenizer, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_data_path", type=str, default="BEIR/datasets", help="Directory to save and load beir datasets")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--add_qd_prompt", type=bool, help="Add a prompt prefix to Q/D")
    parser.add_argument("--num_negatives", type=int, default=100, help="how many negative examples to return")
    # parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")
    # parser.add_argument("--metric", type=str, default="dot", help="Metric used to compute similarity between two embeddings")
    # parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    # parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")

    parser.add_argument("--use_bf16", type=bool, default=False, help="")
    parser.add_argument("--use_gpu", type=bool, default=True, help="")
    parser.add_argument("--use_faiss", type=bool, default=False, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--main_addr", type=str, default='localhost', help="Main IP address.")
    # parser.add_argument("--main_port", type=str, default=6666, help="Main port (for multi-node SLURM jobs)")

    args, _ = parser.parse_known_args()
    main(args)

