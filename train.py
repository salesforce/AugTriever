# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import functools
import logging
import os
import shutil
import sys
import types
import warnings
import torch

import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.integrations import WandbCallback, rewrite_logs

from src.utils import eval_utils
from src.dataloader.hf_dataloader import load_datasets
from src.dataloader.dpr_dataloader import load_dpr_dataset
from src.dataloader.finetune_dataloader import load_finetune_dataset
from src.dataloader.sbert_dataloader import load_sbert_dataset
from src.dataloader.medi_dataloader import load_medi_dataset

from src.trainer import DenseRetrievalTrainer

from src.utils.training_utils import wandb_setup, wandb_setup_eval, reload_model_from_ckpt
from src.dataloader.data_process import PassageDataCollatorWithPadding
from src.arguments import CustomTrainingArguments, HFTrainingArguments, MoCoArguments
from src.model.moco import MoCo
from src.model.inbatch import InBatch
from src.utils import training_utils

logger = logging.getLogger(__name__)


@record
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((CustomTrainingArguments, HFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, hftraining_args, remaining_strings = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), return_remaining_strings=True)
    else:
        training_args, hftraining_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    warnings.warn('remaining_strings:' + str(remaining_strings))

    moco_args = MoCoArguments().parse()

    # Setup logging
    os.makedirs(hftraining_args.output_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(hftraining_args.local_rank) else logging.WARN,
        handlers=[
            logging.FileHandler(hftraining_args.output_dir+"/train_log.txt", mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
        logger.warning(f"Input arguments: \n\t {' '.join(sys.argv)}")

    if (
        os.path.exists(hftraining_args.output_dir)
        and os.listdir(hftraining_args.output_dir)
    ):
        if hftraining_args.do_train and not hftraining_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({hftraining_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        elif not hftraining_args.overwrite_output_dir and os.path.exists(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin")):
            logger.info("Reloading moco_args from %s",
                        os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))
            _, _, moco_args = torch.load(os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {hftraining_args.local_rank}, device: {hftraining_args.device}, n_gpu: {hftraining_args.n_gpu}"
        + f" distributed training: {bool(hftraining_args.local_rank != -1)}, 16-bits training: {hftraining_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(hftraining_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
        logger.info("*" * 50)
        logger.info("Training/evaluation parameters:\n%s", hftraining_args)
        logger.info("Custom training parameters:\n%s", training_args)
        logger.info("MoCo model parameters:\n%s", moco_args)
        logger.info("*" * 50)

    # Set seed before initializing model.
    set_seed(hftraining_args.seed)

    """""""""""""""""""""
    Load HF configs and tokenizer
    """""""""""""""""""""
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": training_args.cache_dir,
        "hidden_dropout_prob": moco_args.hidden_dropout_prob,
        "attention_probs_dropout_prob": moco_args.attention_probs_dropout_prob,
    }
    hf_config = AutoConfig.from_pretrained(moco_args.model_name_or_path, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(moco_args.model_name_or_path, cache_dir=training_args.cache_dir, use_fast=True)
    if moco_args.arch_type == 'moco':
        model = MoCo(moco_args, hf_config)
    elif moco_args.arch_type == 'inbatch':
        model = InBatch(moco_args, hf_config)
    else:
        raise NotImplementedError(f'Unknown arch type {hf_config.arch_type}')
    # Reload models. Not for resume training (reload training status is not supported yet)
    if training_args.reload_model_from:
        logger.info("Reloading model parameters from: " + training_args.reload_model_from)
        if os.path.isdir(training_args.reload_model_from):
            reload_model_from_ckpt(model, training_args.reload_model_from)
        elif training_args.reload_model_from.startswith('facebook/') or 'spider' in training_args.reload_model_from:
            loaded_model, _ = training_utils.load_model(training_args.reload_model_from, pooling=moco_args.pooling)
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            assert isinstance(model, InBatch)
        else:
            raise Exception('R U sure?')
    # use torch.compile if applicable
    if torch.__version__.startswith('2') and hftraining_args.enable_torch2_compile:
        model = torch.compile(model)

    if hftraining_args.do_train:
        # prepare for data loader
        if training_args.data_type == 'finetune':
            train_dataset, dev_dataset = load_finetune_dataset(tokenizer, training_args, hftraining_args, moco_args)
            train_collator = PassageDataCollatorWithPadding(
                tokenizer,
                batch_size=hftraining_args.train_batch_size,
                padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                max_length=training_args.max_seq_length,
                q_len=training_args.max_q_tokens,
                d_len=training_args.max_d_tokens
            )
        elif training_args.data_type == 'dpr':
            train_dataset, dev_dataset = load_dpr_dataset(tokenizer, training_args, hftraining_args, moco_args)
            train_collator = PassageDataCollatorWithPadding(
                tokenizer,
                batch_size=hftraining_args.train_batch_size,
                padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                max_length=training_args.max_seq_length,
                q_len=training_args.max_q_tokens,
                d_len=training_args.max_d_tokens
            )
        elif training_args.data_type == 'hf':
            train_dataset, dev_dataset = load_datasets(tokenizer, training_args, hftraining_args, moco_args)
            train_collator = PassageDataCollatorWithPadding(
                tokenizer,
                batch_size=hftraining_args.train_batch_size,
                padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                max_length=training_args.max_seq_length,
                q_len=training_args.max_q_tokens,
                d_len=training_args.max_d_tokens
            )
        elif training_args.data_type == 'sbert':
            train_dataset, dev_dataset = load_sbert_dataset(tokenizer, training_args, hftraining_args)
            train_collator = PassageDataCollatorWithPadding(
                tokenizer,
                batch_size=hftraining_args.train_batch_size,
                padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                max_length=training_args.max_seq_length,
                q_len=training_args.max_q_tokens,
                d_len=training_args.max_d_tokens
            )
        elif training_args.data_type == 'medi':
            train_dataset, dev_dataset = load_medi_dataset(tokenizer, training_args, hftraining_args)
            train_collator = PassageDataCollatorWithPadding(
                tokenizer,
                batch_size=hftraining_args.train_batch_size,
                padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                max_length=training_args.max_seq_length,
                q_len=training_args.max_q_tokens,
                d_len=training_args.max_d_tokens
            )
        else:
            raise NotImplementedError(f'Not supported data_type {training_args.data_type}. Please choose among [hf/dpr]')

        """""""""""""""""""""
        Set up trainer
        """""""""""""""""""""
        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            print(f"Initializing trainer")

        trainer = DenseRetrievalTrainer(
            model=model,
            args=hftraining_args,
            train_dataset=train_dataset if hftraining_args.do_train else None,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=train_collator
        )
        trainer.training_args = training_args
        trainer.hftraining_args = hftraining_args
        trainer.moco_args = moco_args
        trainer.best_score = -10000.0
        early_stop_patience = sys.maxsize
        trainer.early_stop_patience = early_stop_patience
        trainer.early_stop_counter = 0
        setattr(trainer, 'model_data_training_args', [training_args, hftraining_args, moco_args])
        torch.save(trainer.model_data_training_args, os.path.join(hftraining_args.output_dir, "model_data_training_args.bin"))
        # override wandb setup to log customized hyperparameters
        wandb_callbacks = [ch for ch in trainer.callback_handler.callbacks if isinstance(ch, WandbCallback)]
        wandb_callback = wandb_callbacks[0] if wandb_callbacks else None

        """""""""""""""""""""
        Start Training
        """""""""""""""""""""
        if hftraining_args.process_index == 0 and wandb_callback:
            # remove previous wandb outputs
            if os.path.exists(hftraining_args.output_dir+'/wandb'):
                shutil.rmtree(hftraining_args.output_dir+'/wandb')
                os.makedirs(hftraining_args.output_dir+'/wandb')
            # override wandb_callback's setup method to record our customized hyperparameters
            new_setup = functools.partial(wandb_setup,
                                          hftraining_args=hftraining_args,
                                          training_args=training_args,
                                          moco_args=moco_args, resume=False)
            wandb_callback.setup = types.MethodType(new_setup, wandb_callback)

        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            print(f"Start training")
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(hftraining_args.output_dir, "train_results.txt")
        if hftraining_args.process_index == 0:
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(hftraining_args.output_dir, "trainer_state.json"))

    """""""""""""""""""""
    Start Evaluation
    """""""""""""""""""""
    results = {}
    if hftraining_args.do_eval:
        logger.info("*** Evaluate ***")
        model = model.to(hftraining_args.local_rank)
        model = model.eval()
        wandb_callback = None
        if hftraining_args.process_index == 0 and os.getenv("WANDB_API_KEY", False) and not os.getenv("WANDB_DISABLED"):
            wandb_callback = WandbCallback()
            wandb_setup_eval(wandb_callback, hftraining_args)
        # eval BEIR
        if not training_args.skip_beireval:
            if training_args.beir_datasets is not None:
                final_beir_datasets = training_args.beir_datasets
            else:
                final_beir_datasets = ['msmarco', 'dbpedia-entity', 'fever', 'climate-fever', 'nq', 'hotpotqa',
                                       'quora', 'cqadupstack', 'trec-covid', 'arguana', 'webis-touche2020',
                                       'scidocs', 'scifact', 'nfcorpus', 'fiqa',
                                       'bioasq', 'signal1m', 'trec-news', 'robust04'
                                       ]
            if hftraining_args.process_index == 0:
                try:
                    prev_eval_dir = os.path.join(hftraining_args.output_dir, 'eval_output', 'checkpoint-%d' % (hftraining_args.max_steps))
                    logger.info("Attempt to copy previous beir output: %s" % (prev_eval_dir))
                    shutil.copytree(prev_eval_dir, hftraining_args.output_dir+'/beir_output', dirs_exist_ok=True)
                except Exception as e:
                    logger.info("Error, failed to copy previous beir output: %s (exist? %s) : %s" % (prev_eval_dir, os.path.exists(prev_eval_dir), e.strerror))
            prev_results_beir = []
            try:
                prev_eval_dir = os.path.join(hftraining_args.output_dir, 'eval_output', 'checkpoint-%d' % (hftraining_args.max_steps))
                if os.path.exists(prev_eval_dir):
                    prev_dones = [f[:-5] for f in os.listdir(prev_eval_dir) if f.endswith('.json')]
                    final_beir_datasets = [dataset for dataset in final_beir_datasets if dataset not in prev_dones]
                    prev_results_beir, prev_dones = eval_utils.load_prev_beir_results(hftraining_args.output_dir + '/beir_output')
                    final_beir_datasets = [dataset for dataset in final_beir_datasets if dataset not in prev_dones]
                    logger.info("Found previous beir output, new BEIR datasets: %s" % (str(final_beir_datasets)))
                else:
                    logger.info("Not Found previous beir output, run all: %s" % (str(final_beir_datasets)))
            except OSError as e:
                logger.info("Error, didn't find previous beir output: %s (exist? %s) : %s" % (prev_eval_dir, os.path.exists(prev_eval_dir), str(e)))
                raise e
            except Exception as e:
                logger.info(e)
                raise e
            results_beir = {}
            if len(final_beir_datasets) > 0:
                results_beir = eval_utils.evaluate_beir(
                    model, tokenizer,
                    beir_path=training_args.beir_path,
                    sim_function=model.sim_metric,
                    add_qd_prompt=(training_args.dq_prompt_ratio > 0.0),
                    batch_size=training_args.beir_batch_size,
                    beir_datasets=final_beir_datasets,
                    output_dir=hftraining_args.output_dir+'/beir_output',
                )
            results_beir.update(prev_results_beir)
            results.update(results_beir)
        # eval SentEval
        if not training_args.skip_senteval:
            results_senteval = eval_utils.evaluate_senteval(model, tokenizer,
                                                            output_dir=hftraining_args.output_dir + '/senteval_output',
                                                            eval_senteval_sts_all=True,
                                                            eval_senteval_transfer=True,
                                                            )
            results.update(results_senteval)
        # eval QA
        if not training_args.skip_qaeval:
            embed_dir = os.path.join(hftraining_args.output_dir, 'wiki_emb')
            passages = eval_utils.generate_passage_embeddings(model, tokenizer,
                                                              passage_path=training_args.wiki_passage_path,
                                                              save_dir=embed_dir,
                                                              per_gpu_batch_size=training_args.qa_batch_size)
            if hftraining_args.process_index == 0:
                results_qa = eval_utils.evaluate_qa(model, tokenizer,
                                                    passages=passages,
                                                    passage_path=training_args.wiki_passage_path,
                                                    qa_datasets_path=training_args.qa_datasets_path,
                                                    passages_embeddings_path=embed_dir + '/*',
                                                    encode_batch_size=training_args.qa_batch_size,
                                                    search_batch_size=1, num_workers=32,
                                                    output_dir=hftraining_args.output_dir+'/qa_output'
                                                    )
                results.update(results_qa)
        # post scores to wandb
        if hftraining_args.process_index == 0:
            if wandb_callback and wandb_callback._wandb.run:
                results = rewrite_logs(results)
                if hftraining_args.max_steps:
                    results['train/global_step'] = hftraining_args.max_steps
                wandb_callback._wandb.log({**results})
            output_eval_file = os.path.join(hftraining_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
