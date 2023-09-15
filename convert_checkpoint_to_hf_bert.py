# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import argparse
import src.utils.training_utils as training_utils
from model.inbatch import InBatch
from model.moco import MoCo


def main(args):
    print(f"Loading model..")
    model, tokenizer = training_utils.load_model(args.model_name_or_path)
    if isinstance(model, MoCo):
        assert args.model_type == "shared"
    if isinstance(model, InBatch) and args.model_type == "shared":
        assert id(model.encoder_q.model) == id(model.encoder_k.model)
    print(f"Loading saved state dict into the new model...")
    if args.model_type == "question":
        bert_model = model.encoder_q.model
    elif args.model_type == "context":
        bert_model = model.encoder_k.model
    elif args.model_type == "shared":
        bert_model = model.encoder_q.model
    else:
        raise ValueError
    print(f"Saving the new model to {args.output_dir}")
    bert_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done saving...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_dir", required=False, type=str)
    parser.add_argument("--model_type", type=str, choices=["shared", "question", "context"], default="shared")

    args = parser.parse_args()

    main(args)
