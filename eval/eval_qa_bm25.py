#!/usr/bin/env python3
# Code adapted from Spider (https://github.com/oriram/spider/).

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""
 Command line tool to get dense results and validate them
"""

import argparse
import logging
import os
import time
from typing import List, Tuple

from pyserini.search import SimpleSearcher

from retriever_utils import get_datasets, load_passages, validate, save_results

from src.qa import qa_validation

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.json"


class SparseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        index_name,
        num_threads
    ):
        self.searcher = SimpleSearcher.from_prebuilt_index(index_name)
        self.num_threads = num_threads

    def get_top_docs(
        self, questions: List[str], top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        qids = [str(x) for x in range(len(questions))]
        hits = self.searcher.batch_search(queries=questions, qids=qids, k=top_docs, threads=self.num_threads)
        time1 = time.time()
        logger.info(f"Index search time: {time1 - time0} sec.")
        results = []
        for qid in qids:
            example_hits = hits[qid]
            example_top_docs = [hit.docid for hit in example_hits]
            example_scores = [hit.score for hit in example_hits]
            results.append((example_top_docs, example_scores))
        logger.info(f"Results conversion time: {time.time() - time1} sec.")
        return results


def main(args):
    config = vars(args)

    # get questions & answers
    passages = load_passages(args.ctx_file)
    id2doc = {d['id']: d for d in passages}
    if len(passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    # Create or load retriever
    if args.pyserini_cache is not None:
        os.environ["PYSERINI_CACHE"] = args.pyserini_cache
    retriever = SparseRetriever(args.index_name, args.num_threads)

    # get top k results
    qa_file_dict = get_datasets(args.qa_file)
    for dataset_name, (questions, question_answers) in qa_file_dict.items():
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        if os.path.exists(out_file):
            logger.info(f"Skipping dataset '{dataset_name}' as it already exists")
            continue
        os.makedirs(dataset_output_dir, exist_ok=True)
        top_ids_and_scores = retriever.get_top_docs(questions, args.n_docs)

        # compute scores
        match_type = "regex" if "curated" in dataset_name else "string"
        match_stats = qa_validation.calculate_matches(id2doc, question_answers, top_ids_and_scores, args.validation_workers, match_type)

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
            passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
            output_no_text=args.output_no_text
        )
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

    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--n-docs", type=int, default=100, help="Amount of top docs to return"
    )
    parser.add_argument("--output_no_text", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="wikipedia-dpr"
    )
    parser.add_argument(
        "--pyserini_cache",
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assert not os.path.exists(os.path.join(args.output_dir, RECALL_FILE_NAME))
    assert not os.path.exists(os.path.join(args.output_dir, RESULTS_FILE_NAME))
    main(args)
