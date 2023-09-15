# Code adapted from Transformers (https://github.com/huggingface/transformers) governed by Apache-2.0 license.

# Copyright (c) 2023, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed, IntervalStrategy,
)
import argparse
import os

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict, Tuple
from transformers.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

@dataclass
class ModelArguments:
    dummy: Optional[str] = field(default=None, metadata={"help": "for back compatibility."})
@dataclass
class ExtHFTrainingArguments(TrainingArguments):
    dummy: Optional[str] = field(default=None, metadata={"help": "for back compatibility."})


@dataclass
class CustomTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    cache_dir: Optional[str] = field(default=None, metadata={
        "help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    reload_model_from: Optional[str] = field(default=None, metadata={"help": "If set, will load and replace the parameters of current model."})
    # dataset arguments
    train_file: Optional[str] = field(default=None, metadata={"help": "The training data file (.txt or .csv)."})
    train_prob: Optional[str] = field(default=None, metadata={"help": "The sampling probability for multiple datasets."})
    dev_file: Optional[str] = field(default=None, metadata={"help": "The dev data file (.txt or .csv)."})
    # eval
    skip_beireval: bool = field(default=False, metadata={"help": ""})
    skip_senteval: bool = field(default=False, metadata={"help": ""})
    skip_qaeval: bool = field(default=False, metadata={"help": ""})
    # BEIR eval
    beir_path: Optional[str] = field(default="/export/home/data/beir", metadata={ "help": "Base directory of BEIR data."})
    beir_datasets: List[str] = field(default=None, metadata={"help": "Specify what BEIR datasets will be used in evaluation."
                    "Only affect the do_test phrase, not effective for during-training evaluation."})
    beir_batch_size: int = field(default=128, metadata={"help": "Specify batch size for BEIR evaluation."})
    # QA eval
    wiki_passage_path: Optional[str] = field(default="/export/home/data/search/nq/psgs_w100.tsv", metadata={ "help": "Base directory of wiki data (DRP version)."})
    qa_datasets_path: Optional[str] = field(default="/export/home/data/search/nq/qas/*-test.csv,/export/home/data/search/nq/qas/entityqs/test/P*.test.json", metadata={ "help": "QA datasets (glob pattern)."})
    qa_eval_steps: int = field(default=-1, metadata={"help": "Specify step frequency for QA evaluation."})
    qa_batch_size: int = field(default=128, metadata={"help": "Specify batch size for QA evaluation."})
    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"})
    # parameters used in data transformation (data_process.hfdataset_prepare_features)
    data_type: str = field(default=None, metadata={"help": "Specify which data loader will be used for training, hf/dpr."})
    data_pipeline_name: str = field(default=None, metadata={"help": "Pre-defined data pipeline name. If set, all data hyper-parameters below will be overwritten."})
    pseudo_query_names: str = field(default=None, metadata={"help": "Specify the names of augmented queries and probs, e.g. {'title':0.2,'T03B-topic':0.8}."})
    resume_training: str = field(default=None, metadata={"help": "resume training."})
    max_context_len: int = field(default=None, metadata={"help": "if data_type is document and max_context_len is given, we first randomly crop a contiguous span, and Q/D will be sampled from it."})
    min_dq_len: int = field(default=None, metadata={"help": "The minimal number of words for sampled query and doc."})
    min_q_len: float = field(default=None, metadata={"help": "min Query len. If less 1.0, it denotes a length ratio."})
    max_q_len: float = field(default=None, metadata={"help": "max Query len. If less 1.0, it denotes a length ratio."})
    min_d_len: float = field(default=None, metadata={"help": "min Doc len. If less 1.0, it denotes a length ratio."})
    max_d_len: float = field(default=None, metadata={"help": "max Doc len. If less 1.0, it denotes a length ratio."})
    word_del_ratio: float = field(default=0.0, metadata={"help": "Ratio for applying word deletion, for both Q and D."})
    query_in_doc: bool = field(default=False, metadata={"help": "Whether sampled query must appear in doc. "})
    dq_prompt_ratio: float = field(default=0.0, metadata={"help": "Randomly add a prefix to indicate the input is Q/D."})
    title_as_query_ratio: float = field(default=0.0, metadata={"help": "randomly use title as query."})
    include_title_ratio: float = field(default=0.0, metadata={"help": "whether doc title is added (at the beginning)."})
    # parameters used in tokenization (data_process.PassageDataCollatorWithPadding)
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."})
    max_q_tokens: List[int] = field(default=None, metadata={"help": "Can be an int or a range, specify the max length of q, and d_len=max_seq_length-q_len"})
    max_d_tokens: List[int] = field(default=None, metadata={"help": "max d_len"})
    mlm_probability: float = field(default=0.15, metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"})
    # fine-tuning
    finetune: bool = field(default=False, metadata={"help": "Is it a fine-tuning dataset or pretraining?"})
    negative_strategy: str = field(default='random', metadata={"help": "random/first/multiple, specify the way to return negatives"})
    hard_negative_ratio: float = field(default=0.0, metadata={"help": "Ratio of hard negatives during training, the rest are randomly sampled."})
    hard_negative_num: int = field(default=-1, metadata={"help": "How many hard negative examples to be considered for sampling, -1 means all."})


    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class HFTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(default=False, metadata={"help": "Evaluate transfer task dev sets (in validation)."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use."},)
    num_cycles: float = field(default=0.5, metadata={"help": "."})
    lr_end: float = field(default=1e-7, metadata={"help": "."})
    power: float = field(default=1.0, metadata={"help": "."})
    enable_torch2_compile: bool = field(default=False, metadata={"help": "use PyTorch2 feature torch.compile."})


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MoCoArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument("--arch_type", type=str, default='moco', help="moco or inbatch")
        self.parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased', help="backbone")
        self.parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
        self.parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)

        self.parser.add_argument('--indep_encoder_k', type=str2bool, default=False, help='whether to use an independent/asynchronous encoder.')
        self.parser.add_argument('--projection_size', type=int, default=768)
        self.parser.add_argument("--num_q_view", type=int, default=1)
        self.parser.add_argument("--num_k_view", type=int, default=1)
        self.parser.add_argument("--q_proj", type=str, default='none', help="Q projector MLP setting, format is 1.`` or none: no projecter; 2.mlp: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE); 3. 1024-2048: three dense layers (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)")
        self.parser.add_argument("--k_proj", type=str, default='none', help="D projector MLP setting. Same as --q_proj, except for `shared` means reusing q_proj layer.")

        self.parser.add_argument('--num_random_chunk', type=int, default=0, help="number of random chunks as query candidates")
        self.parser.add_argument("--q_extract", type=str, default=None, help="Strategy for query selection, options: [self-dot]")
        self.parser.add_argument("--q_extract_ratio", type=float, default=None, help="p% of queries are model-selected chunks, the rest use given queries")

        self.parser.add_argument("--queue_strategy", type=str, default='fifo', help="'fifo', 'priority'")
        self.parser.add_argument("--num_extra_pos", type=int, default=0)
        self.parser.add_argument("--neg_names", type=str, nargs='+', default=None, help='specify the key names of negative data.')
        self.parser.add_argument("--use_inbatch_negatives", type=str2bool, default=False, help='whether to include negative data in current batch for loss')
        self.parser.add_argument("--queue_size", type=int, default=0)
        self.parser.add_argument("--q_queue_size", type=int, default=0)
        self.parser.add_argument("--symmetric_loss", type=str2bool, default=False)
        self.parser.add_argument("--sim_metric", type=str, default='dot', help='What similarity metric function to use (dot, cosine).')
        self.parser.add_argument('--pooling', type=str, default='average', help='average or cls')
        self.parser.add_argument("--pooling_dropout", type=str, default='none', help="none, standard, gaussian, variational")
        self.parser.add_argument("--pooling_dropout_prob", type=float, default=0.0, help="bernoulli, gaussian, variational")
        self.parser.add_argument('--merger_type', type=str, default=None, help="projector MLP setting, format is "
           "(1)`none`: no projecter; (2) multiview; (3)`mlp`: a simple D by D dense layer with Tanh activation, no parameter sharing (used in SimCSE) "
           "(4) `1024-2048`: multi-layer dense connections (D*1024*2048) with BatchNorm1d and ReLU (barlow-twin)")
        self.parser.add_argument("--warmup_queue_size_ratio", type=float, default=0.0, help='linearly increase queue size to 100% until training_steps*warmup_queuesize_ratio.')
        self.parser.add_argument("--num_warmup_stage", type=int, default=None, help='.')
        self.parser.add_argument("--queue_update_steps", type=int, default=1, help='we only update the model parameters (backprop) every k step, and but update queue in the rest k-1 steps.')
        self.parser.add_argument("--momentum", type=float, default=0.9995)
        self.parser.add_argument("--temperature", type=float, default=0.05)
        self.parser.add_argument('--label_smoothing', type=float, default=0.)
        self.parser.add_argument('--norm_query', action='store_true')
        self.parser.add_argument('--norm_doc', action='store_true')
        self.parser.add_argument('--moco_train_mode_encoder_k', action='store_true')
        self.parser.add_argument('--random_init', action='store_true', help='init model with random weights')
        # q/k diff regularizer
        self.parser.add_argument('--qk_norm_diff_lambda', type=float, default=0.0)
        # alignment+uniformity
        self.parser.add_argument('--align_unif_loss', type=str2bool, default=False)
        self.parser.add_argument('--align_unif_cancel_step', type=int, default=-1)
        self.parser.add_argument('--align_weight', type=float, default=0.0)
        self.parser.add_argument('--align_alpha', type=float, default=2)
        self.parser.add_argument('--unif_weight', type=float, default=0.0)
        self.parser.add_argument('--unif_t', type=float, default=2)

    def print_options(self, opt):
        message = ''
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: %s]' % str(default)
            message += f'{str(k):>40}: {str(v):<40}{comment}\n'
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, 'models'))
        file_name = os.path.join(opt.output_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
