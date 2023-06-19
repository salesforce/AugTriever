import os
import json
import tqdm
import numpy as np
import torch
import argparse

from datasets import Dataset
from typing import List, Dict
from functools import partial
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader
from src.mteb import MTEB, AbsTaskRetrieval, DRESModel

from src.mtebeval.utils import pool, logger, move_to_cuda

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='bert-base-uncased',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--data-dir', default='tmp-datasets/',
                    type=str, metavar='N', help='dataset directory')
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--corpus-chunk-size', type=int, default=5000, help='corpus chunk size')




args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg'], 'pool_type should be cls or avg'
assert args.output_dir, 'output_dir should be set'
os.makedirs(args.output_dir, exist_ok=True)


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    return tokenizer(examples['contents'],
                     max_length=512,
                     padding=True,
                     return_token_type_ids=False,
                     truncation=True)


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()
        # if self.gpu_count > 1:
        #     self.encoder = torch.nn.DataParallel(self.encoder)

        self.model = self  # AbsTaskRetrieval, L132, start_multi_process_pool() needs it...
        self.sep = '. '
        self.encoder.cuda()
        self.encoder.eval()

    def encode(self, input_texts: List[str], **kwargs) -> np.ndarray:
        inputs = self.tokenizer(input_texts)
        batch_dict = move_to_cuda(inputs)
        encoded_embeds = []

        with torch.cuda.amp.autocast():
            outputs: BaseModelOutput = self.encoder(**batch_dict)
            embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
            encoded_embeds.append(embeds.cpu())
        return torch.cat(encoded_embeds, axis=0)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['query: {}'.format(q) for q in queries]
        return self._do_encode(input_texts, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['passage: {}'.format(t) for t in input_texts]
        return self._do_encode(input_texts, **kwargs)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str],
                   convert_to_tensor: bool=False,
                   batch_size: int=128, **kwargs) -> np.ndarray:
        dataset: Dataset = Dataset.from_dict({'contents': input_texts})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if convert_to_tensor:
                    encoded_embeds.append(embeds.cpu())
                else:
                    encoded_embeds.append(embeds.cpu().numpy())

        if convert_to_tensor:
            return torch.cat(encoded_embeds, axis=0)
        else:
            return np.concatenate(encoded_embeds, axis=0)


def main():
    assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)
    model = RetrievalModel()

    task_names = [t.description["name"] for t in MTEB(task_types=['Retrieval'], task_langs=['en']).tasks]
    logger.info('Tasks: {}'.format(task_names))

    # task_names = ['SCIDOCS', 'SciFact']
    for task in task_names:
        logger.info('Processing task: {}'.format(task))

        args.doc_as_query = task in ['QuoraRetrieval']
        # if task in ['MSMARCOv2']:
        #     logger.warning('Skip task: {}, since it has no test split'.format(task))
        #     continue

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test" if task not in ['MSMARCO', 'MSMARCOv2'] else 'dev'],
                       output_folder=args.output_dir,
                       data_folder=args.data_dir,
                       batch_size=args.batch_size,
                       overwrite_results=False)


if __name__ == '__main__':
    main()
