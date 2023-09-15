import argparse

import torch
import os
from transformers import AutoConfig, AutoTokenizer
from transformers.models.dpr import DPRQuestionEncoder, DPRContextEncoder

PREFIX_NAME_DICT = {
    "shared": "model.",
    "question": "question_model.",
    "context": "ctx_model."
}


def load_states_from_checkpoint(ckpt_path):
    state_dict = torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location="cpu")
    # remove queue
    keys_to_remove = [k for k in list(state_dict.keys()) if k.startswith('queue') or k.startswith('encoder_k')]
    for k in keys_to_remove:
        del state_dict[k]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder_q'):
            new_k = k.replace('encoder_q.model.', '')
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def main(args):
    print(f"Loading config from {args.config_name}")
    config = AutoConfig.from_pretrained(args.config_name)
    print(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"Loading model..")
    if args.model_type == "question":
        model = DPRQuestionEncoder(config)
    else:
        model = DPRContextEncoder(config)
    print(f"Model class: {type(model)}")

    print(f"Loading saved state from pickle at: {args.ckpt_path}")
    print("Reloading moco_args from %s", os.path.join(args.ckpt_path, "model_data_training_args.bin"))
    _, _, moco_args = torch.load(os.path.join(args.ckpt_path, "model_data_training_args.bin"))
    saved_model_state_dict = load_states_from_checkpoint(args.ckpt_path)

    prefix = PREFIX_NAME_DICT[args.model_type]
    print(f"Model type: {args.model_type}. Looking for prefix {prefix}")

    # assert len(saved_model_state_dict) == len(model.state_dict())

    print(f"Loading saved state dict into the new model...")
    if hasattr(model, "ctx_encoder"):
        assert args.model_type in ["shared", "context"]
        model.ctx_encoder.bert_model.load_state_dict(saved_model_state_dict)
    elif hasattr(model, "question_encoder"):
        assert args.model_type in ["shared", "question"]
        model.question_encoder.bert_model.load_state_dict(saved_model_state_dict)
    else:
        raise ValueError

    print(f"Saving the new model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done saving...")

    if args.hf_model_name is not None:
        print(f"Uploading the model to {args.hf_model_name}. Private? {args.hf_private_model}")
        model.push_to_hub(args.hf_model_name, private=args.hf_private_model)
        tokenizer.push_to_hub(args.hf_model_name, private=args.hf_private_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--model_type", type=str, choices=["shared", "question", "context"], default="shared")

    # Configuration args:
    parser.add_argument("--config_name", default="facebook/dpr-ctx_encoder-single-nq-base", type=str)
    parser.add_argument("--tokenizer_name", default="facebook/dpr-ctx_encoder-single-nq-base", type=str)

    # HF upload args:
    parser.add_argument("--hf_model_name", type=str, default=None)
    parser.add_argument("--hf_private_model", action="store_true")

    args = parser.parse_args()

    main(args)
