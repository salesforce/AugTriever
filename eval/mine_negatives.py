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
import numpy as np
import logging
import json
import os
# os.environ['OMP_NUM_THREADS'] = str(32)

from tqdm import tqdm

import src.beireval.slurm as slurm
import src.beireval.beir_utils as beir_utils
import src.utils.training_utils as training_utils
import src.beireval.dist_utils as dist_utils
from src.beir.datasets.data_loader import GenericDataLoader
from src.beir.retrieval.evaluation import EvaluateRetrieval
from src.beir.util import download_and_unzip
from beir.retrieval.search.lexical import BM25Search as BM25

from src.beir.retrieval.search.dense import DenseRetrievalExactSearch, FlatIPFaissSearch

import torch.distributed as dist

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def setup(args):
    slurm.init_distributed_mode(args)
    slurm.init_signal_handler()
    os.makedirs(args.output_dir, exist_ok=True)

    logger = training_utils.setup_logger()
    logger.info(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = training_utils.load_model(args.model_name_or_path)

    if args.use_gpu:
        model = model.cuda()
        model = model.half()
    return model, tokenizer


def mine_msmarco_dense_model(args, tokenizer, model):
    '''
    # os.environ['OMP_NUM_THREADS'] = 1
    https://github.com/facebookresearch/faiss/issues/2502
    no, it doesn't matter...
    '''
    args.dataset = 'msmarco'
    # args.dataset = 'nq'
    # args.dataset = 'trec-covid'
    # args.dataset = 'nfcorpus'
    # args.dataset = 'scifact'
    logger.info(f"Start indexing with dataset={args.dataset}")
    split = 'train' if args.dataset == 'msmarco' else 'test'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = download_and_unzip(url, args.beir_data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    if dist.is_initialized():
        logger.info(f'device={dist.get_rank()}, #(corpus)={len(corpus)}, #(queries)={len(queries)}, #(qrels)={len(qrels)}')
    else:
        logger.info(f'#(corpus)={len(corpus)}, #(queries)={len(queries)}')
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
            use_gpu=True,  # can speed up 1000x than on cpu
            add_qd_prompt=args.add_qd_prompt,
            corpus_chunk_size=8192
        )
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            dmodel.index(corpus, metric)
            dmodel.save(args.output_dir, args.dataset, split)
            dmodel.load(args.output_dir, args.dataset, split)
        if dist.is_initialized():
            dist.barrier()
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
    retriever = EvaluateRetrieval(dmodel, score_function=metric, k_values=[100])
    predicts = retriever.retrieve(corpus, queries)
    # load again to remove prompts
    # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    for docid, ctx in corpus.items():
        ctx['passage_id'] = docid
    if dist_utils.is_main():
        ndcg, _map, recall, precision = retriever.evaluate(qrels, predicts, k_values=[5, 10, 100])
        output_file = f'{args.output_dir}/{args.dataset}.jsonl'
        logger.info(f'Dumping negatives to {output_file}')
        export_beir_to_dpr_format(output_file, args.num_negatives, corpus, queries, qrels, predicts,
                                  dataset_name=f'{args.dataset}-{split}')


def export_msmarco_no_negative(args):
    args.dataset = 'msmarco'
    split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = download_and_unzip(url, args.beir_data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    logger.info(f'Dumping negatives to {args.output_dir}/{args.dataset}.jsonl')
    progress_bar = tqdm(range(len(qrels)), desc=f"Creating DPR formatted {args.dataset} file")
    with open(f'{args.output_dir}/{args.dataset}.jsonl', 'w') as fp:
        for cnt, (query_id, pos_doc2score) in enumerate(qrels.items()):
            # query
            query = queries[query_id]
            # positive doc
            pos_doc_id, pos_score = list(pos_doc2score.items())[0]
            pos_ctx = corpus[pos_doc_id]
            pos_ctx['passage_id'] = pos_doc_id
            pos_ctx['score'] = pos_score
            json.dump({"id": query_id,
                       "question": query,
                       "answers": [],
                       "positive_ctxs": [pos_ctx],
                       "hard_negative_ctxs": []}, # empty negatives
                      fp)
            fp.write("\n")
            progress_bar.update(1)


def export_msmarco_random_negatives(args):
    args.dataset = 'msmarco'
    split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = download_and_unzip(url, args.beir_data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    for docid, ctx in corpus.items():
        ctx['passage_id'] = docid
    all_docs = list(corpus.values())

    logger.info(f'Dumping data to {args.output_dir}/{args.dataset}-random{args.num_negatives}.jsonl')
    progress_bar = tqdm(range(len(qrels)), desc=f"Creating DPR formatted {args.dataset} file")
    with open(f'{args.output_dir}/{args.dataset}-random{args.num_negatives}.jsonl', 'w') as fp:
        for cnt, (query_id, pos_doc2score) in enumerate(qrels.items()):
            # query
            query = queries[query_id]
            # positive doc
            pos_docid, pos_score = list(pos_doc2score.items())[0]
            pos_ctx = corpus[pos_docid]
            pos_ctx['passage_id'] = pos_docid
            pos_ctx['score'] = pos_score
            # random negative docs
            neg_idxs = np.random.randint(0, len(all_docs), size=args.num_negatives)
            neg_ctxs = [all_docs[i] for i in neg_idxs if all_docs[i]['passage_id'] != pos_docid]
            json.dump({
                "dataset": f'{args.dataset}-{split}',
                "question_id": query_id,
                "question": query,
                "answers": [],
                "positive_ctxs": [pos_ctx],
                "negative_ctxs": neg_ctxs,
                "hard_negative_ctxs": []
               }, fp)
            fp.write("\n")
            progress_bar.update(1)


def mine_msmarco_bm25(args):
    args.dataset = 'msmarco'
    # args.dataset = 'scifact'
    split = 'train'
    hostname = "http://localhost:9200"  # localhost
    index_name = f"bm25-{args.dataset}-train"
    initialize = False  # False if load and use existing index

    logger.info(f'Loading data')
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    data_path = download_and_unzip(url, args.beir_data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    for docid, ctx in corpus.items():
        ctx['passage_id'] = docid

    logger.info(f'#doc={len(corpus)}, #query={len(queries)}, #qrels={len(qrels)}')
    logger.info(f'Start retrieving w/ BM25')
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model, k_values=[args.num_negatives])
    predicts = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, predicts, [10, 100])
    output_file = f'{args.output_dir}/{args.dataset}-bm25.jsonl'
    export_beir_to_dpr_format(output_file, args.num_negatives, corpus, queries, qrels, predicts, dataset_name=f'{args.dataset}-{split}')


def mine_msmarco_exact(args, tokenizer, model):
    '''
    very slow, MS-MARCO can take ~24h
    '''
    args.dataset = 'msmarco'
    # args.dataset = 'scifact'
    logger.info(f"Start indexing with dataset={args.dataset}")
    split = 'train'

    output_dict = beir_utils.evaluate_model(
        query_encoder=model,
        doc_encoder=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        batch_size=args.per_gpu_batch_size,
        query_batch_size=args.per_gpu_batch_size,
        norm_query=model.norm_query,
        norm_doc=model.norm_doc,
        is_main=dist_utils.is_main(),
        split=split,
        metric=model.sim_metric,
        beir_data_path=args.beir_data_path,
        add_qd_prompt=args.add_qd_prompt,
        corpus_chunk_size=8192,
        return_all=True,
        k_values=[100]
    )
    ndcg, _map, recall, precision, mrr, recall_cap, hole = output_dict['scores']
    corpus = output_dict['corpus']
    queries = output_dict['queries']
    qrels = output_dict['qrels']
    predicts = output_dict['predicts']
    for docid, ctx in corpus.items():
        ctx['passage_id'] = docid

    if dist_utils.is_main():
        output_file = f'{args.output_dir}/{args.dataset}.jsonl'
        export_beir_to_dpr_format(output_file, args.num_negatives, corpus, queries, qrels, predicts, dataset_name=f'{args.dataset}-{split}')


def export_beir_to_dpr_format(output_path, num_negatives, corpus, queries, qrels, predicts, dataset_name):
    logger.info(f'Dumping negatives to {output_path}')
    progress_bar = tqdm(range(len(qrels)), desc=f"Exporting...")
    all_docs = list(corpus.values())
    with open(output_path, 'w') as fp:
        for cnt, (query_id, pos_doc2score) in enumerate(qrels.items()):
            if query_id not in predicts:  continue  # skip erroneous cases
            # query
            query = queries[query_id]
            # positive doc
            pos_docid, pos_score = list(pos_doc2score.items())[0]
            pos_ctx = corpus[pos_docid]
            pos_ctx['passage_id'] = pos_docid
            pos_ctx['score'] = pos_score
            # random negative docs
            neg_idxs = np.random.randint(0, len(all_docs), size=args.num_negatives)
            neg_ctxs = [all_docs[i] for i in neg_idxs if all_docs[i]['passage_id'] != pos_docid]
            # hard negative docs
            hard_neg_ctxs = []
            pred_d2scores = sorted(predicts[query_id].items(), key=lambda k: k[1], reverse=True)
            for neg_docid, score in pred_d2scores[:num_negatives]:
                if neg_docid == pos_docid: continue
                neg_ctx = corpus[neg_docid]
                neg_ctx['passage_id'] = neg_docid
                neg_ctx['score'] = score
                hard_neg_ctxs.append(neg_ctx)
            json.dump({
                "dataset": dataset_name,
                "question_id": query_id,
                "question": query,
                "answers": [],
                "positive_ctxs": [pos_ctx],
                "negative_ctxs": neg_ctxs,
                "hard_negative_ctxs": hard_neg_ctxs,
               }, fp)
            fp.write("\n")
            progress_bar.update(1)
    logger.info(f'Done')


def mine_nq(args, tokenizer, model):
    pass


def main(args):
    # model, tokenizer = setup(args)
    # mine_msmarco_dense_model(args, tokenizer, model)

    export_msmarco_random_negatives(args)
    # mine_msmarco_bm25(args)
    # mine_nq(args, tokenizer, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_data_path", type=str, default="BEIR/datasets", help="Directory to save and load beir datasets")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--add_qd_prompt", type=bool, default=False, help="Add a prompt prefix to Q/D")
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

