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
import pickle
import torch
import tqdm

import src.beireval.slurm as slurm
import src.model.baseencoder
import src.utils.training_utils as training_utils
import src.qa.data
import src.qa.normalize_text as normalize_text

import logging
logger = logging.getLogger(__name__)

def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in tqdm.tqdm(enumerate(passages), desc='Encoding passages'):
            batch_ids.append(p["id"])
            if args.no_title or not "title" in p:
                text = p["text"]
            else:
                text = p["title"] + " " + p["text"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                if 'sent_emb' in inspect.getfullargspec(model.forward).args:
                    # Ours
                    emb = model(**encoded_batch, sent_emb=True, is_query=False).pooler_output
                else:
                    # Contriever or other HFTransformer models
                    ids, mask = encoded_batch['input_ids'], encoded_batch['attention_mask']
                    ids, mask = ids.cuda(), mask.cuda()
                    emb = model(ids, mask)
                if hasattr(emb, 'pooler_output'):  # HFTransformer models
                    emb = emb['pooler_output']

                emb = emb.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(emb)

                batch_text = []
                batch_ids = []
                if k % 1000 == 0 and k > 0:
                    logger.info(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    logger.info(f"Model loaded from {args.model_name_or_path}.", flush=True)
    d_model, tokenizer = training_utils.load_model(args.model_name_or_path)
    d_model.eval()
    d_model = d_model.cuda()
    if not args.no_fp16:
        d_model = d_model.half()

    passages = src.qa.data.load_dpr_passages(args.passages)

    num_shards = int(os.environ['WORLD_SIZE'])
    shard_id = args.local_rank
    shard_size = len(passages) // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size
    if shard_id == num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    logger.info(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, d_model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f"_{shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    logger.info(f"Total passages processed {len(allids)}. Written to {save_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="dir path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="prefix path to save embeddings")
    parser.add_argument("--local_rank", type=int, default=0, help="Id of the current device")

    # parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    # parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")

    args = parser.parse_args()

    slurm.init_distributed_mode(args)

    main(args)

