# Code adapted from Transformers (https://github.com/huggingface/transformers) governed by Apache-2.0 license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import contextlib
import os
import sys
import numpy as np
import logging
import torch
from typing import Union

from src.utils import dist_utils
from src.arguments import MoCoArguments
from src.model.inbatch import InBatch
from src.model.moco import MoCo
from src.utils.model_utils import load_encoder

Number = Union[float, int]
from transformers import is_torch_tpu_available, AutoModel, AutoTokenizer, DPRContextEncoder


def wandb_setup_eval(self, args, resume=True):
    run_name = args.run_name
    wandb_project_name = os.getenv("WANDB_PROJECT", "huggingface")
    run_id = os.getenv("WANDB_RUN_ID", None)
    if self._wandb.run is None:
        if run_id:
            resume = 'must'
        self._wandb.init(
            project=wandb_project_name,
            name=run_name,
            id=run_id,
            resume=resume
        )
    # define default x-axis (for latest wandb versions)
    if getattr(self._wandb, "define_metric", None):
        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)


def wandb_setup(cls, args=None, state=None, model=None, hftraining_args=None, model_args=None, training_args=None, moco_args=None, resume=False, **kwargs):
    """
    Modified based on WandbCallback at L534 of transformers.integration
    to keep track of our customized parameters (moodel_args, data_args)
    """
    if cls._wandb is None:
        return
    cls._initialized = True
    if state.is_world_process_zero:
        combined_dict = {**args.to_sanitized_dict()}

        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        if hftraining_args is not None:
            hftraining_args = hftraining_args.to_dict()
            combined_dict = {**hftraining_args, **combined_dict}
        if model_args is not None:
            model_args = model_args.to_dict()
            combined_dict = {**model_args, **combined_dict}
        if training_args is not None:
            training_args = training_args.to_dict()
            combined_dict = {**training_args, **combined_dict}
        if moco_args is not None:
            moco_args = vars(moco_args)
            combined_dict = {**moco_args, **combined_dict}

        trial_name = state.trial_name
        init_args = {}
        if trial_name is not None:
            run_name = trial_name
            init_args["group"] = args.run_name
        else:
            run_name = args.run_name
        init_args['resume'] = resume

        if cls._wandb.run is None:
            cls._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                name=run_name,
                settings=cls._wandb.Settings(code_dir="."), # for code saving
                **init_args,
            )
        # add config parameters (run may have been created manually)
        cls._wandb.config.update(combined_dict, allow_val_change=True)

        # define default x-axis (for latest wandb versions)
        if getattr(cls._wandb, "define_metric", None):
            cls._wandb.define_metric("train/global_step")
            cls._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

        # keep track of model topology and gradients, unsupported on TPU
        if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
            cls._wandb.watch(
                model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
            )

        cls._wandb.run.log_code(".")


def setup_logger(logger_name='root', output_dir=None, stdout_only=False):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if dist_utils.is_main() else logging.WARN)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    handlers = [stdout_handler]
    if not stdout_only and output_dir:
        file_handler = logging.FileHandler(filename=os.path.join(output_dir, 'run.log'))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    for hdl in handlers:
        logger.addHandler(hdl)
    return logger


def get_parameters(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = "[Network] Total number of parameters : %.6f M" % (num_params / 1e6)
    return message

def get_inbatch_base_config():
    inbatch_args = MoCoArguments().parse()
    inbatch_args.arch_type = 'inbatch'

    return inbatch_args

def get_moco_base_config():
    moco_args = MoCoArguments().parse()
    return moco_args


def load_model(model_name_or_path, pooling=None):
    if os.path.exists(model_name_or_path):
        if os.path.exists(os.path.join(model_name_or_path, "model_data_training_args.bin")):
            # original AugTriever checkpoints
            args = torch.load(os.path.join(model_name_or_path, "model_data_training_args.bin"))
            training_args, hf_config, moco_args = args
            if pooling:
                moco_args.pooling = pooling
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            if hasattr(training_args, 'arch_type'):
                arch_type = training_args.arch_type
            elif hasattr(moco_args, 'arch_type'):
                arch_type = moco_args.arch_type
            else:
                print('arch_type is not found, use moco')
                arch_type = 'moco'
            if arch_type == 'moco':
                model = MoCo(moco_args, hf_config)
            elif arch_type == 'inbatch':
                model = InBatch(moco_args)
            else:
                raise NotImplementedError(f'Unknown arch type {arch_type}')
            model = reload_model_from_ckpt(model, model_name_or_path)
        else:
            # a converted huggingface DPR checkpoint
            model_args = get_inbatch_base_config()
            model_args.model_name_or_path = 'bert-base-uncased'
            if pooling:
                model_args.pooling = pooling
            else:
                model_args.pooling = 'average'
            model = InBatch(model_args)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
            ckpt = DPRContextEncoder.from_pretrained(model_name_or_path,
                                                     use_auth_token='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')
            state_dict = ckpt.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                nk = k.replace('ctx_encoder.bert_model.', 'encoder_q.model.')
                new_state_dict[nk] = v
                nk = k.replace('ctx_encoder.bert_model.', 'encoder_k.model.')
                new_state_dict[nk] = v
            model.load_state_dict(new_state_dict, strict=True)
    elif model_name_or_path.startswith('facebook'):
        model_args = get_inbatch_base_config()
        model_args.model_name_or_path = model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if pooling:
            model_args.pooling = pooling
        else:
            if model_name_or_path.startswith('facebook/contriever'):
                # e.g. facebook/contriever
                model_args.pooling = 'average'
            elif model_name_or_path.startswith('facebook/spar'):
                # facebook/spar-wiki-bm25-lexmodel-query-encoder
                model_args.pooling = 'cls'
            else:
                raise NotImplementedError('Doublecheck!')
        model = InBatch(model_args)
        model = reload_model_from_pretrained(model, model_name_or_path, strict=False)
    elif 'spider' in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_args = get_inbatch_base_config()
        model_args.model_name_or_path = 'bert-base-uncased'
        if pooling:
            model_args.pooling = pooling
        else:
            model_args.pooling = 'cls'
        model = InBatch(model_args)
        ckpt = DPRContextEncoder.from_pretrained(model_name_or_path)
        state_dict = ckpt.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            nk = k.replace('ctx_encoder.bert_model.', 'encoder_q.model.')
            new_state_dict[nk] = v
            nk = k.replace('ctx_encoder.bert_model.', 'encoder_k.model.')
            new_state_dict[nk] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        # e.g. 'sentence-transformers/all-mpnet-base-v2' and other Huggingface models
        model, tokenizer = load_encoder(model_name_or_path, pooling=pooling)
    # else:
    #     # BERT etc.
    #     moco_args = get_moco_base_config()
    #     hf_config = AutoConfig.from_pretrained(moco_args.model_name_or_path)
    #     tokenizer = AutoTokenizer.from_pretrained(moco_args.model_name_or_path, use_fast=True)
    #     model = MoCo(moco_args, hf_config)
    #     model = reload_model_from_pretrained(model, moco_args.model_name_or_path)

    return model, tokenizer


def reload_model_from_pretrained(model, model_name, strict=True):
    ckpt = AutoModel.from_pretrained(model_name)
    state_dict = ckpt.state_dict()
    del state_dict['pooler.dense.weight']
    del state_dict['pooler.dense.bias']
    items = list(state_dict.items())
    for k, v in items:
        nk = 'encoder_q.model.' + k
        state_dict[nk] = v
        nk = 'encoder_k.model.' + k
        state_dict[nk] = v
        del state_dict[k]
    model.load_state_dict(state_dict, strict=strict)
    return model


def reload_model_from_ckpt(model, ckpt_path):
    state_dict = torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location="cpu")
    # back compatibility: for loading old ckpts
    if 'encoder_q.model.embeddings.word_embeddings.weight' not in state_dict:
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
    if isinstance(model, InBatch):  # MoCo->InBatch
        queue_keys = [k for k in list(state_dict.keys()) if k.startswith('queue')]
        for k in queue_keys:
            del state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys + unexpected_keys) > 0:
        print('!' * 50)
        print('Found not matched keys while loading checkpoint from: ', ckpt_path)
        print('missing_keys: ', str(missing_keys))
        print('unexpected_keys: ', str(unexpected_keys))
        print('!' * 50)
    return model


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
