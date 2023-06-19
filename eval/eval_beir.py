# Code adapted from Contriever (https://github.com/facebookresearch/contriever/)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import src.utils.training_utils as training_utils
logger = training_utils.setup_logger()

import argparse
import json
import os
import logging
from pathlib import Path

import torch
import numpy as np
import src.beireval.beir_utils as beir_utils

import src.beireval.slurm as slurm
import src.utils.dist_utils as dist_utils

BEIR_datasets = [
        'trec-covid', 'nfcorpus', 'nq', 'hotpotqa',
        'cqadupstack', 'quora',
        'fiqa', 'arguana', 'webis-touche2020',
        'msmarco', 'signal1m', 'trec-news',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact',
        'bioasq', 'robust04',
        ]
BEIR_public_datasets = [
        'trec-covid', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact',
        'cqadupstack', 'quora',
        'msmarco'
        ]
small_datasets = ['fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack']


def main(args):
    slurm.init_distributed_mode(args)
    slurm.init_signal_handler()
    assert args.metric
    assert args.pooling
    logger.setLevel(logging.INFO if dist_utils.is_main() else logging.WARN)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading model from [{args.model_name_or_path}]")

    q_model, tokenizer = training_utils.load_model(args.model_name_or_path, pooling=args.pooling)
    if args.doc_model_name_or_path is not None:
        d_model, _ = training_utils.load_model(args.doc_model_name_or_path)
    else:
        d_model = q_model
    q_model = q_model.eval()
    d_model = d_model.eval()
    if torch.cuda.is_available():
        q_model = q_model.cuda()
        d_model = d_model.cuda()
    if torch.__version__.startswith('2'):
        q_model = torch.compile(q_model)
        d_model = torch.compile(d_model)

    logger.info(f"Start indexing with dataset=[{args.dataset}]")

    if args.dataset == 'all':
        datasets = BEIR_datasets
    elif args.dataset == 'public':
        datasets = BEIR_public_datasets
    elif args.dataset == 'small':
        datasets = small_datasets
    else:
        assert args.dataset in BEIR_datasets, f'Unknown dataset [{args.dataset}], supported datasets: \n {str(BEIR_datasets)}'
        datasets = [args.dataset]

    metrics = {}
    avg_ndcg_10, avg_recall_10, avg_recall_20, avg_recall_100 = [], [], [], []
    for dataset in datasets:
        split = 'dev' if dataset == 'msmarco' else 'test'
        logger.info(f"Start evaluating with dataset=[{dataset}], split=[{split}]")
        if os.path.exists(f"{args.output_dir}/{dataset}.json"):
            logger.info(f"Found previous results, skip evaluating [{dataset}]")
            continue
        ndcg, _map, recall, precision, mrr, recall_cap, hole, mtebtask_results = beir_utils.evaluate_model(
            query_encoder=q_model,
            doc_encoder=d_model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=args.per_gpu_batch_size,
            norm_query=args.norm_query,
            norm_doc=args.norm_doc,
            is_main=dist_utils.is_main(),
            split=split,
            metric=args.metric,
            beir_data_path=args.beir_data_path,
            add_qd_prompt=False,
            corpus_chunk_size=20480
        )

        if dist_utils.is_main():
            ndcg10 = ndcg['NDCG@10']
            recall10 = recall['Recall@10'] if dataset != 'trec-covid' else recall_cap['R_cap@10']
            recall20 = recall['Recall@20'] if dataset != 'trec-covid' else recall_cap['R_cap@20']
            recall100 = recall['Recall@100'] if dataset != 'trec-covid' else recall_cap['R_cap@100']
            metrics[f'eval_beir-{dataset}_ndcg@10'] = ndcg10
            metrics[f'eval_beir-{dataset}_recall@10'] = recall10
            metrics[f'eval_beir-{dataset}_recall@20'] = recall20
            metrics[f'eval_beir-{dataset}_recall@100'] = recall100
            avg_ndcg_10.append(ndcg10)
            avg_recall_10.append(recall10)
            avg_recall_20.append(recall20)
            avg_recall_100.append(recall100)

            result_dict = {
                'dataset': dataset,
                'split': split,
                'metric': args.metric,
                'norm_query': args.norm_query,
                'norm_doc': args.norm_doc,
                'scores': {
                    'ndcg': ndcg,
                    'map': _map,
                    'precision': precision,
                    'recall': recall,
                    'mrr': mrr,
                    'recall_cap': recall_cap,
                    'hole': hole,
                }
            }
            logger.info(f"Dump results of {dataset} to {args.output_dir}/{dataset}.json")
            print(result_dict)
            with open(f"{args.output_dir}/{dataset}.json", 'w') as writer:
                writer.write(json.dumps(result_dict, indent=4) + "\n")
            rows = ['metric,@1,@3,@5,@10,@20,@50,@100,@200,@1000']
            for metric_name, scores in result_dict['scores'].items():
                row = ','.join([str(s) for s in ([metric_name] + list(scores.values()))])
                rows.append(row)
            with open(f"{args.output_dir}/{dataset}.csv", 'w') as writer:
                for row in rows:
                    writer.write(row + "\n")
            # export MTEB scores
            if args.mteb_output_dir is None:
                mteb_output_dir = os.path.join(Path(args.output_dir).parent.absolute(), 'mteb')
            else:
                mteb_output_dir = args.mteb_output_dir
            os.makedirs(mteb_output_dir, exist_ok=True)
            for mtebtask_result in mtebtask_results:
                logger.info(f"Dump MTEB results of {dataset} to {os.path.join(mteb_output_dir, mtebtask_result['mteb_dataset_name'] + '.json')}")
                with open(os.path.join(mteb_output_dir, mtebtask_result['mteb_dataset_name'] + '.json'), "w") as f_out:
                    json.dump(mtebtask_result, f_out, indent=2, sort_keys=True)

            metrics['eval_beir-avg_ndcg@10'] = np.mean(avg_ndcg_10)
            metrics['eval_beir-avg_recall@10'] = np.mean(avg_recall_10)
            metrics['eval_beir-avg_recall@20'] = np.mean(avg_recall_20)
            metrics['eval_beir-avg_recall@100'] = np.mean(avg_recall_100)

            print(metrics)
    print(f'Rank:{dist_utils.get_rank()} finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_data_path", type=str, default="BEIR/datasets", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--mteb_output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Query model name or path")
    parser.add_argument("--doc_model_name_or_path", type=str, default=None, help="Doc model name or path")
    parser.add_argument("--pooling", type=str, default=None, help="average or cls")
    parser.add_argument("--metric", type=str, default=None, help="Metric used to compute similarity between two embeddings, dot or cosine")
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_addr", type=str, default=-1, help="Main IP address.")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
    parser.add_argument("--compile_model", action="store_true", help="Enable the PyTorch2 feature, model.compile()")

    args, _ = parser.parse_known_args()
    main(args)
