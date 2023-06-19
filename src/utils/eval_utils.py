# Code adapted from Contriever (https://github.com/facebookresearch/contriever/),
# Spider (https://github.com/oriram/spider/) and SimCSE (https://github.com/princeton-nlp/SimCSE).

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import csv
import glob
import json
import os
import pickle

import time
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict

from src.qa.index import Indexer
from src.qa.qa_validation import calculate_matches
from src.beireval import beir_utils
from src.utils import dist_utils
from src.qa.data import load_dpr_passages, get_qa_datasets
from src.qa.normalize_text import normalize
from src.senteval import engine
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Set path to SentEval
PATH_TO_DATA = '/export/share/ruimeng/project/search/simcse/SentEval/data/'
RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.jsonl"


def load_prev_beir_results(beir_output_dir):
    if dist.is_initialized():
        dist.barrier()
    time.sleep(np.random.uniform(0.0, 3.0))  # to avoid reading error when there are multiple workers
    datasets = []
    metrics = {}
    if not os.path.exists(beir_output_dir): return metrics, datasets
    json_files = [os.path.join(beir_output_dir, f) for f in os.listdir(beir_output_dir) if f.endswith('.json') and not f.startswith('senteval')]
    if len(json_files) == 0: return {}

    for fp in json_files:
        try:
            score_dict = json.load(open(fp, 'r'))
        except Exception as e:
            print(f'Error while loading, will try again: {fp}')
            time.sleep(np.random.uniform(0.0, 3.0))  # to avoid reading error when there are multiple workers
            try:
                score_dict = json.load(open(fp, 'r'))
            except Exception as e:
                print(f'Error again, skip')
                # raise e
        if 'dataset' not in score_dict: continue  # skip beir.json
        dataset = score_dict['dataset']
        datasets.append(dataset)
        score_dict = score_dict['scores']
        ndcg10 = score_dict['ndcg']['NDCG@10']
        # recall10 = score_dict['recall']['Recall@10']
        # recall20 = score_dict['recall']['Recall@20']
        recall100 = score_dict['recall']['Recall@100']
        metrics[f'eval_beir-{dataset}_ndcg@10'] = ndcg10
        # metrics[f'eval_beir-{dataset}_recall@10'] = recall10
        # metrics[f'eval_beir-{dataset}_recall@20'] = recall20
        metrics[f'eval_beir-{dataset}_recall@100'] = recall100

    return metrics, datasets


def embed_passages(passages, model, tokenizer,
                   lowercase=True, normalize_text=True, passage_maxlength=512,
                   per_gpu_batch_size=128):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])
            text = p["title"] + " " + p["text"]
            if lowercase:
                text = text.lower()
            if normalize_text:
                text = normalize(text)
            batch_text.append(text)

            if len(batch_text) == per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch, sent_emb=True, is_query=False).pooler_output

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 1000 == 0 and k > 0:
                    logger.info(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def embed_queries(queries, model, tokenizer,
                  lowercase=True, normalize_text=True,
                  question_maxlength=512, per_gpu_batch_size=128
                  ):
    model.eval()
    query_vectors, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if lowercase:
                q = q.lower()
            if normalize_text:
                q = normalize(q)
            batch_question.append(q)

            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch, sent_emb=True, is_query=True)
                output = output.pooler_output
                if output.is_cuda:
                    output = output.cpu()
                query_vectors.append(output)
                batch_question = []

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info(f"device={dist.get_rank()}, questions embeddings shape: {query_tensor.size()}")
    assert query_tensor.size(0) == len(queries)
    return query_tensor.numpy()


def generate_passage_embeddings(model, tokenizer, passage_path, save_dir, per_gpu_batch_size):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model = model.cuda()
    shard_id = dist.get_rank()
    save_path = save_dir + f"/emb_{shard_id:02d}"
    if os.path.exists(save_path):
        logger.info(f"device={shard_id},embedding file exists, skip generation! ({save_path}).")
        return None
    passages = load_dpr_passages(passage_path)

    num_shards = int(os.environ['WORLD_SIZE'])
    shard_size = len(passages) // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size
    if shard_id == num_shards - 1:
        end_idx = len(passages)

    logger.info(f"device={shard_id}, embedding generation for {len(passages)} passages, shard-{shard_id} from idx {start_idx} to {end_idx}.")
    passage_shard = passages[start_idx:end_idx]
    allids, allembeddings = embed_passages(passage_shard, model, tokenizer, per_gpu_batch_size=per_gpu_batch_size)
    logger.info(f"device={shard_id}, saving {len(allids)} passage embeddings to {save_path}.")
    with open(save_path, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)
    logger.info(f"Total passages processed {len(allids)}. Written to {save_path}.")
    if dist.is_initialized():
        dist.barrier()
    return passages


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        if not os.path.isfile(file_path): continue
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    logger.info(f"Device {dist.get_rank()}, data indexing completed, allembeddings.shape={allembeddings.shape}.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    logger.info(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    elif data_path.endswith(".csv"):
        with open(data_path, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            data = []
            for k, row in enumerate(reader):
                ex = {"question": row[0], "answers": row[1]}
                data.append(ex)
    return data


def save_results(
    id2doc: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    output_no_text: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [id2doc[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
        hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
        ctxs_num = len(hits)

        d = {
                "question": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        # "title": docs[c]['title'],
                        # "text": docs[c]['text'] if not output_no_text else "",  # to save space
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        merged_data.append(d)

    if out_file.endswith('json'):
        with open(out_file, "w") as writer:
            writer.write(json.dumps(merged_data, indent=4) + "\n")
    else:
        with open(out_file, "w") as writer:
            for d in merged_data:
                writer.write(json.dumps(d) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def evaluate_qa(model, tokenizer,
                passages, passage_path, qa_datasets_path,
                passages_embeddings_path, output_dir,
                encode_batch_size=128, search_batch_size=8,
                num_workers=16, n_docs=100,
                ):
    if os.path.exists(f"{output_dir}/qa.json"):
        logger.info(f"Trying to load prev senteval results: {output_dir}/qa.json")
        try:
            with open(f"{output_dir}/qa.json", 'r') as reader:
                metrics = json.load(reader)
                if 'eval_qa-nq-test-acc@5' in metrics:
                    logger.info(f"Success! Skip running QA and return prev results!")
                    return metrics
                else:
                    logger.info(f"Failure! Not valid prev results. Rerun QA-eval!")
        except Exception as e:
            logger.info(f"Failure! Rerun QA-eval!")
            pass
    if passages is None:
        passages = load_dpr_passages(passage_path)
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(model, DistributedDataParallel):
        model = model.module
    index = Indexer(model.projection_size, num_threads=num_workers)

    input_paths = glob.glob(passages_embeddings_path)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0]) if len(input_paths) > 0 else os.path.dirname(passages_embeddings_path)
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if os.path.exists(index_path):
        logger.info(f"Loading previous index from {index_path}")
        index.deserialize_from(embeddings_dir)
    else:
        # index all passages
        os.makedirs(embeddings_dir, exist_ok=True)
        logger.info(f"Indexing passages from files {input_paths}")
        start_time = time.time()
        index_encoded_data(index, input_paths, encode_batch_size)
        logger.info(f"Indexing time: {time.time()-start_time:.1f}s")
        index.serialize(embeddings_dir)
        # reload index to free memory (otherwise it's easy to oom. strange! )
        del index
        torch.cuda.empty_cache()
        index = Indexer(model.projection_size, num_threads=num_workers)
        index.deserialize_from(embeddings_dir)

    logger.info("Moving index to GPUs")
    start_time = time.time()
    index.to_gpu()
    logger.info(f"Moving index to GPUs time: {time.time()-start_time:.1f}s")

    eq_score_dict = defaultdict(list)
    score_dict = {}
    # load passages
    id2doc = {d['id']: d for d in passages}
    # get questions & answers
    qa_file_dict = get_qa_datasets(qa_datasets_path)
    for dataset_name, (questions, question_answers) in qa_file_dict.items():
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        # encode questions
        questions_embedding = embed_queries(questions, model, tokenizer,
                                            lowercase=True, normalize_text=True,
                                            question_maxlength=512,
                                            per_gpu_batch_size=encode_batch_size)
        # get top k results
        start_time = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, n_docs, index_batch_size=search_batch_size)
        logger.info(f"Search time: {time.time()-start_time:.1f} s.")

        # compute scores
        start_time = time.time()
        match_type = "regex" if "curated" in dataset_name else "string"
        match_stats = calculate_matches(id2doc, question_answers, top_ids_and_scores, num_workers, match_type)
        logger.info(f"Match time: {time.time()-start_time:.1f} s.")
        top_k_hits = match_stats.top_k_hits
        logger.info("Validation results: top k documents hits %s", top_k_hits)
        top_k_hits = [v / len(top_ids_and_scores) for v in top_k_hits]
        logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        logger.info(f"Saved recall@k info to {out_file}")
        with open(out_file, "w") as f:
            for k, recall in enumerate(top_k_hits):
                f.write(f"{k + 1},{recall}\n")
        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        save_results(
            id2doc,
            questions,
            question_answers,
            top_ids_and_scores,
            match_stats.questions_doc_hits,
            out_file
        )
        if dataset_name.startswith('P'):
            eq_score_dict["entityqs-acc@5"].append(top_k_hits[4])
            eq_score_dict["entityqs-acc@20"].append(top_k_hits[19])
            eq_score_dict["entityqs-acc@100"].append(top_k_hits[-1])
        else:
            score_dict[f"eval_qa-{dataset_name}-acc@5"] = top_k_hits[4]
            score_dict[f"eval_qa-{dataset_name}-acc@20"] = top_k_hits[19]
            score_dict[f"eval_qa-{dataset_name}-acc@100"] = top_k_hits[-1]
        logger.info("{}, acc@5={:.4f}, acc@20={:.4f}, acc@100={:.4f}".format(dataset_name, top_k_hits[4], top_k_hits[19], top_k_hits[-1]))
    if len(eq_score_dict) > 0:
        assert len(eq_score_dict["entityqs-acc@5"]) == 24
        score_dict["eval_qa-entityqs-macro-acc@5"] = np.mean(eq_score_dict["entityqs-acc@5"])
        score_dict["eval_qa-entityqs-macro-acc@20"] = np.mean(eq_score_dict["entityqs-acc@20"])
        score_dict["eval_qa-entityqs-macro-acc@100"] = np.mean(eq_score_dict["entityqs-acc@100"])

    with open(f"{output_dir}/qa.json", 'w') as writer:
        writer.write(json.dumps(score_dict, indent=4) + "\n")

    return score_dict


def evaluate_beir(model, tokenizer, beir_path, output_dir, sim_function, add_qd_prompt=False, batch_size=32, beir_datasets=None) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    if not beir_datasets:
        # fever will cause gpu error when `Encoding Batch 88/109`
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact'] # quick test
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'quora', 'dbpedia-entity', 'nq'] # mostly reported in Contriever
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid', 'nq', 'dbpedia-entity', 'quora'] # small testsets+NQ+FEVER+Quora
        beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack',
                         'trec-covid', 'quora', 'nq']  # smallest 8 datasets+quora,nq
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid']  # smallest 8 datasets
        # beir_datasets = ['fiqa']  # test
    if isinstance(model, DistributedDataParallel):
        model = model.module
    norm_query = model.norm_query
    norm_doc = model.norm_doc
    beir_data_path = beir_path

    metrics = {}
    avg_ndcg_10 = []
    avg_recall_10 = []
    avg_recall_20 = []
    avg_recall_100 = []

    for dataset in beir_datasets:
        if dist.is_initialized():
            dist.barrier()
        logger.info(f"Start evaluating with dataset={dataset}")
        split = 'dev' if dataset == 'msmarco' else 'test'
        ndcg, _map, recall, precision, mrr, recall_cap, hole, _ = beir_utils.evaluate_model(
            query_encoder=model,
            doc_encoder=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            norm_query=norm_query,
            norm_doc=norm_doc,
            is_main=dist_utils.is_main(),
            split=split,
            metric=sim_function,
            beir_data_path=beir_data_path,
            add_qd_prompt=add_qd_prompt,
            corpus_chunk_size=20480
        )

        if dist_utils.is_main():
            # logger.info(dataset + ' ' + str(ndcg))
            # logger.info(dataset + ' ' + str(_map))
            # logger.info(dataset + ' ' + str(recall))
            # logger.info(dataset + ' ' + str(precision))
            # logger.info(dataset + ' ' + str(mrr))
            # logger.info(dataset + ' ' + str(recall_cap))
            # logger.info(dataset + ' ' + str(hole))
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
                'metric': sim_function,
                'norm_query': norm_query,
                'norm_doc': norm_doc,
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
            logger.info(f"Dump results of {dataset} to {output_dir}/{dataset}.json")
            with open(f"{output_dir}/{dataset}.json", 'w') as writer:
                writer.write(json.dumps(result_dict, indent=4) + "\n")
            rows = ['metric,@1,@3,@5,@10,@20,@50,@100,@200,@1000']
            for metric_name, scores in result_dict['scores'].items():
                row = ','.join([str(s) for s in ([metric_name] + list(scores.values()))])
                rows.append(row)
            with open(f"{output_dir}/{dataset}.csv", 'w') as writer:
                for row in rows:
                    writer.write(row + "\n")

    metrics['eval_beir-avg_ndcg@10'] = np.mean(avg_ndcg_10)
    metrics['eval_beir-avg_recall@10'] = np.mean(avg_recall_10)
    metrics['eval_beir-avg_recall@20'] = np.mean(avg_recall_20)
    metrics['eval_beir-avg_recall@100'] = np.mean(avg_recall_100)

    with open(f"{output_dir}/beir.json", 'w') as writer:
        writer.write(json.dumps(metrics, indent=4) + "\n")

    return metrics


def evaluate_senteval(model, tokenizer, output_dir,
                      eval_senteval_sts_all: bool = False,
                      eval_senteval_transfer: bool = False) -> Dict[str, float]:
    if dist.is_initialized():
        dist.barrier()
    if os.path.exists(f"{output_dir}/senteval-core.json"):
        logger.info(f"Trying to load prev senteval results: {output_dir}/senteval-core.json")
        time.sleep(np.random.uniform(0.0, 5.0))  # to avoid reading error when there are multiple workers
        try:
            with open(f"{output_dir}/senteval-core.json", 'r') as reader:
                metrics = json.load(reader)
                if 'eval_senteval-avg_transfer' in metrics:
                    logger.info(f"Success! Skip running senteval and return prev results!")
                    return metrics
                else:
                    logger.info(f"Failure! Not valid prev results. Rerun senteval!")
        except Exception as e:
            logger.info(f"Failure! rerun senteval!")
            pass
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].cuda()
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            pooler_output = outputs.pooler_output
        return pooler_output.cpu()

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    if eval_senteval_transfer:
        tasks = tasks + ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    if eval_senteval_sts_all:
        tasks = tasks + ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

    model.eval()
    results = se.eval(tasks)

    if eval_senteval_sts_all:
        metrics = {}
        avg_sts = 0
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            avg_sts += results[task]['all']['spearman']['all']
            metrics['eval_senteval-{}'.format(task)] = results[task]['all']['spearman']['all']
        avg_sts /= 7
        metrics['eval_senteval-avg_sts_7'] = avg_sts
    else:
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        metrics = {"eval_senteval-stsb_spearman": stsb_spearman, "eval_senteval-sickr_spearman": sickr_spearman,
                   "eval_senteval-avg_sts": (stsb_spearman + sickr_spearman) / 2}

    if eval_senteval_transfer:
        avg_transfer = 0
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            avg_transfer += results[task]['devacc']
            metrics['eval_senteval-{}'.format(task)] = results[task]['devacc']
        avg_transfer /= 7
        metrics['eval_senteval-avg_transfer'] = avg_transfer

    results.update(metrics)

    logger.info(f"SentEval time: {time.time() - start_time:.1f}s")
    logger.info(f"SentEval scores: \n\t\t{str(metrics)}")
    with open(f"{output_dir}/senteval.json", 'w') as writer:
        writer.write(json.dumps(results, indent=4) + "\n")
    with open(f"{output_dir}/senteval-core.json", 'w') as writer:
        writer.write(json.dumps(metrics, indent=4) + "\n")
    if dist.is_initialized():
        dist.barrier()

    return metrics
