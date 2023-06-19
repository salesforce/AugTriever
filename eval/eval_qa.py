# Code adapted from Spider (https://github.com/oriram/spider/).

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import inspect
import argparse
import logging
import pickle
import time
import glob

import numpy as np
import torch

import src.qa.index
import src.model.baseencoder
import src.beireval.slurm
import src.qa.data
from src.utils import training_utils
from src.qa.qa_validation import calculate_matches
import src.qa.normalize_text
from src.qa.data import load_dpr_passages
from src.utils.eval_utils import save_results, RECALL_FILE_NAME, RESULTS_FILE_NAME

os.environ['OMP_NUM_THREADS'] = '32'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    query_vectors, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                if 'sent_emb' in inspect.getfullargspec(model.forward).args:
                    # Ours
                    output = model(**encoded_batch, sent_emb=True, is_query=False).pooler_output
                else:
                    # Contriever or other HFTransformer models
                    ids, mask = encoded_batch['input_ids'], encoded_batch['attention_mask']
                    ids, mask = ids.cuda(), mask.cuda()
                    output = model(ids, mask)
                if hasattr(output, 'pooler_output'):  # HFTransformer models
                    output = output['pooler_output']
                output = output.cpu()
                query_vectors.append(output)
                batch_question = []

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info(f"Questions embeddings shape: {query_tensor.size()}")
    assert query_tensor.size(0) == len(queries)
    return query_tensor.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    logger.info("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def main(args):
    logger.info(f"Loading model from: {args.model_name_or_path}")
    q_model, tokenizer = training_utils.load_model(args.model_name_or_path)

    q_model.eval()
    q_model = q_model.cuda()
    if not args.no_fp16:
        q_model = q_model.half()

    index = src.qa.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits, args.num_threads)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f}s")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)
    if not args.faiss_cpu:
        logger.info("Moving index to GPUs")
        start_time_retrieval = time.time()
        index.to_gpu()
        logger.info(f"Moving index to GPUs time: {time.time()-start_time_retrieval:.1f}s")

    # load passages
    start_time_retrieval = time.time()
    passages = load_dpr_passages(args.passage_path)
    id2doc = {d['id']: d for d in passages}

    logger.info(f"Loading passages time: {time.time()-start_time_retrieval:.1f}s")

    # get questions & answers
    qa_file_dict = src.qa.data.get_qa_datasets(args.qa_file)
    for dataset_name, (questions, question_answers) in qa_file_dict.items():
        # init
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        if os.path.exists(dataset_output_dir):
            logger.info(f"Skipping dataset '{dataset_name}' as it already exists")
            logger.info(f"{dataset_output_dir}")
            continue
        os.makedirs(dataset_output_dir, exist_ok=True)

        # encode questions
        questions_embedding = embed_queries(args, questions, q_model, tokenizer)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs, index_batch_size=1)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        # compute scores
        match_type = "regex" if "curated" in dataset_name else "string"
        match_stats = calculate_matches(id2doc, question_answers, top_ids_and_scores, args.validation_workers, match_type)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--qa_file", required=True, type=str, default=None,
        help=".json file containing question and answers, similar format to reader data")
    parser.add_argument("--passage_path", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix")
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument( "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--model_name_or_path", type=str, help="path to directory containing model weights and config file")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--faiss_cpu", action="store_true", help="otherwise gpu")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers", type=int, default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--num_threads", type=int, default=64, help="Number of threads to use while searching in the index")

    args = parser.parse_args()
    src.beireval.slurm.init_distributed_mode(args)
    main(args)
