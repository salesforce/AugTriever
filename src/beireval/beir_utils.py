import logging
logger = logging.getLogger(__name__)

import os
import inspect

import torch
import torch.distributed as dist
from typing import List, Dict
import numpy as np

import src.mteb as mteb
import src.mteb.tasks as mteb_tasks
from src.beir.util import download_and_unzip
from src.beir.datasets.data_loader import GenericDataLoader
from src.beir.retrieval.evaluation import EvaluateRetrieval
from src.beir.retrieval.search.dense import DenseRetrievalExactSearch
from src.beireval.dist_utils import varsize_gather_nograd


beir2mteb_mapping = {
    'msmarco': mteb_tasks.MSMARCO,
    'trec-covid': mteb_tasks.TRECCOVID,
    'nfcorpus': mteb_tasks.NFCorpus,
    'nq': mteb_tasks.NQ,
    'hotpotqa': mteb_tasks.HotpotQA,
    'fiqa': mteb_tasks.FiQA2018,
    'arguana': mteb_tasks.ArguAna,
    'webis-touche2020': mteb_tasks.Touche2020,
    'dbpedia-entity': mteb_tasks.DBPedia,
    'scidocs': mteb_tasks.SCIDOCS,
    'fever': mteb_tasks.FEVER,
    'climate-fever': mteb_tasks.ClimateFEVER,
    'scifact': mteb_tasks.SciFact,
    'quora': mteb_tasks.QuoraRetrieval,
    'cqadupstack-android': mteb_tasks.CQADupstackAndroidRetrieval,
    'cqadupstack-english': mteb_tasks.CQADupstackEnglishRetrieval,
    'cqadupstack-gaming': mteb_tasks.CQADupstackGamingRetrieval,
    'cqadupstack-gis': mteb_tasks.CQADupstackGisRetrieval,
    'cqadupstack-mathematica': mteb_tasks.CQADupstackMathematicaRetrieval,
    'cqadupstack-physics': mteb_tasks.CQADupstackPhysicsRetrieval,
    'cqadupstack-programmers': mteb_tasks.CQADupstackProgrammersRetrieval,
    'cqadupstack-stats': mteb_tasks.CQADupstackStatsRetrieval,
    'cqadupstack-tex': mteb_tasks.CQADupstackTexRetrieval,
    'cqadupstack-unix': mteb_tasks.CQADupstackUnixRetrieval,
    'cqadupstack-webmasters': mteb_tasks.CQADupstackWebmastersRetrieval,
    'cqadupstack-wordpress': mteb_tasks.CQADupstackWordpressRetrieval,
    # not included in MTEB
    'bioasq': 'BioASQ',
    'signal1m': 'Signal-1m',
    'robust04': 'TREC-Robust04',
    'trec-news': 'TREC-News',
}

class DenseEncoderModel:
    def __init__(
        self, 
        query_encoder, 
        doc_encoder=None, 
        tokenizer=None, 
        maxlength=512, 
        add_special_tokens=True, 
        norm_query=False, 
        norm_doc=False,
        **kwargs
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc

    def encode_queries(self, queries: List[str], batch_size: int, use_gpu=False, **kwargs) -> np.ndarray:

        if dist.is_initialized(): 
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))
        queries = [queries[i] for i in idx]

        allemb = []
        nbatch = (len(queries)-1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k+1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx], 
                    max_length=self.maxlength, 
                    padding=True, 
                    truncation=True, 
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt", 
                )
                ids, mask = qencode['input_ids'], qencode['attention_mask']
                ids, mask = ids.cuda(), mask.cuda()

                if 'is_query' in inspect.getfullargspec(self.query_encoder.forward).args:
                    emb = self.query_encoder(input_ids=ids, attention_mask=mask, sent_emb=True, is_query=True)
                elif 'sent_emb' in inspect.getfullargspec(self.query_encoder.forward).args:
                    emb = self.query_encoder(input_ids=ids, attention_mask=mask, sent_emb=True)
                else:  # for some HF models don't have normalize
                    emb = self.query_encoder(ids, mask)
                # # @memray for ANCE
                # if 'is_query' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, is_query=True)
                # # @memray for SimCSE
                # elif 'sent_emb' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, sent_emb=True)
                # # @memray for some HF models don't have normalize
                # elif 'normalize' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, normalize=self.norm_query)
                # else:
                #     emb = self.query_encoder(ids, mask)
                if hasattr(emb, 'pooler_output'):
                    emb = emb['pooler_output']
                allemb.append(emb)

        allemb = torch.cat(allemb, dim=0) 
        if dist.is_initialized():
            allemb = varsize_gather_nograd(allemb)
        if not use_gpu:
            allemb = allemb.cpu().numpy()
        return allemb


    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, return_cpu=True, **kwargs):

        if dist.is_initialized(): 
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        _corpus = [corpus[i] for i in idx]
        _corpus = [
            c['title'] + ' TEXT: ' + c['text'] if len(c['title']) > 0 else c['text'] for c in _corpus
        ]
        
        allemb = []
        nbatch = (len(_corpus)-1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k+1) * batch_size, len(_corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    _corpus[start_idx:end_idx],
                    max_length=self.maxlength, 
                    padding=True, 
                    truncation=True, 
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt", 
                )
                ids, mask = cencode['input_ids'], cencode['attention_mask']
                ids, mask = ids.cuda(), mask.cuda()

                if 'is_query' in inspect.getfullargspec(self.doc_encoder.forward).args:
                    emb = self.doc_encoder(ids, mask, sent_emb=True, is_query=False)
                elif 'sent_emb' in inspect.getfullargspec(self.doc_encoder.forward).args:
                    emb = self.doc_encoder(input_ids=ids, attention_mask=mask, sent_emb=True)
                else:  # for some HF models don't have normalize
                    emb = self.doc_encoder(ids, mask)
                if hasattr(emb, 'pooler_output'):
                    emb = emb['pooler_output']
                allemb.append(emb)

        allemb = torch.cat(allemb, dim=0)
        if dist.is_initialized():
            allemb = varsize_gather_nograd(allemb)
        if return_cpu:
            allemb = allemb.cpu().numpy()
        return allemb


def evaluate_model(
        query_encoder, 
        doc_encoder, 
        tokenizer, 
        dataset, 
        batch_size=128, 
        query_batch_size=128,
        max_length=512,
        add_special_tokens=True,
        norm_query=False, 
        norm_doc=False, 
        is_main=True, 
        split='test', 
        metric='dot',
        beir_data_path="BEIR/datasets",
        add_qd_prompt=False,
        corpus_chunk_size=50000,
        k_values=[1, 3, 5, 10, 20, 50, 100, 200, 1000],
        return_all=False
    ):
    if metric == 'cosine':
        metric = 'cos_sim'
    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder, 
            doc_encoder=doc_encoder, 
            tokenizer=tokenizer,
            maxlength=max_length,
            add_special_tokens=add_special_tokens, 
            norm_query=norm_query, 
            norm_doc=norm_doc,
        ),
        batch_size=batch_size,
        query_batch_size=query_batch_size,
        add_qd_prompt=add_qd_prompt,
        corpus_chunk_size=corpus_chunk_size
    )
    retriever = EvaluateRetrieval(dmodel,
                                  score_function=metric,
                                  k_values=k_values
                                  )
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = download_and_unzip(url, beir_data_path)
    mtebtask_results = []
    if dataset == 'cqadupstack':
        ndcgs, _maps, recalls, precisions, mrrs, recall_caps, holes = [], [], [], [], [], [], []
        cqasubsets = [
            'android', 
            'english', 
            'gaming', 
            'gis', 
            'mathematica', 
            'physics', 
            'programmers', 
            'stats', 
            'tex', 
            'unix', 
            'webmasters', 
            'wordpress'
        ]
        for sub in cqasubsets:
            data_folder = os.path.join(data_path, sub)
            corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split=split)
            if is_main: print(f'Start retrieving {dataset}-{sub}, #(corpus)={len(corpus)}, #(queries)={len(queries)}, '
                              f'batch_size={retriever.retriever.batch_size}, chunk_size={retriever.retriever.corpus_chunk_size}')
            results = retriever.retrieve(corpus, queries)
            if is_main:
                # print(f'Start evaluating, #(qrels)={len(qrels)}, #(results)={len(results)}')
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
                recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
                hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
                ndcgs.append(ndcg)
                _maps.append(_map)
                recalls.append(recall)
                precisions.append(precision)
                mrrs.append(mrr)
                recall_caps.append(recall_cap)
                holes.append(hole)

                # for MTEB
                task = beir2mteb_mapping[f'{dataset}-{sub}']()
                mtebtask_result = {
                    "mteb_version": mteb.__version__,
                    "dataset_revision": task.description.get("revision", None),
                    "mteb_dataset_name": task.description['name'],
                }
                mteb_scores = {
                    **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                    **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                    **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                    **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                    **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
                }
                mtebtask_result[split] = mteb_scores
                mtebtask_results.append(mtebtask_result)
        if is_main:
            # average scores
            print('Dataset: ', dataset)
            ndcg = {key: sum(item.get(key) for item in ndcgs) / 12 for key in ndcgs[0]}
            _map = {key: sum(item.get(key) for item in _maps) / 12 for key in _maps[0]}
            recall = {key: sum(item.get(key) for item in recalls) / 12 for key in recalls[0]}
            precision = {key: sum(item.get(key) for item in precisions) / 12 for key in precisions[0]}
            mrr = {key: sum(item.get(key) for item in mrrs) / 12 for key in mrrs[0]}
            recall_cap = {key: sum(item.get(key) for item in recall_caps) / 12 for key in recall_caps[0]}
            hole = {key: sum(item.get(key) for item in holes) / 12 for key in holes[0]}
        else:
            ndcg, _map, recall, precision = None, None, None, None
            mrr, recall_cap, hole = None, None, None
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        # qrels = {k:v for k,v in list(qrels.items())[:10]}
        # queries = {k:queries[k] for k in qrels.keys()}
        # corpus = {k:v for k,v in list(corpus.items())[:100]}
        if is_main: print(f'Start retrieving, #(corpus)={len(corpus)}, #(queries)={len(queries)},'
                          f'batch_size={retriever.retriever.batch_size}, chunk_size={retriever.retriever.corpus_chunk_size}')
        results = retriever.retrieve(corpus, queries)
        if is_main:
            print(f'Start evaluating {dataset}, #(qrels)={len(qrels)}, #(results)={len(results)}, #(corpus)={len(corpus)}, #(queries)={len(queries)}')
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
            recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
            hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
            # for MTEB
            if not isinstance(beir2mteb_mapping[dataset], str):
                task = beir2mteb_mapping[dataset]()
                dataset_revision = task.description.get("revision", None)
                mteb_dataset_name = task.description['name']
            else:
                dataset_revision = None
                mteb_dataset_name = beir2mteb_mapping[dataset]
            mtebtask_result = {
                "mteb_version": mteb.__version__,
                "dataset_revision": dataset_revision,
                "mteb_dataset_name": mteb_dataset_name,
            }
            mteb_scores = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }
            mtebtask_result[split] = mteb_scores
            mtebtask_results.append(mtebtask_result)
        else:
            ndcg, _map, recall, precision = None, None, None, None
            mrr, recall_cap, hole = None, None, None

    if return_all:
        return {'corpus': corpus,
                'queries': queries,
                'qrels': qrels,
                'predicts': results,
                'scores': (ndcg, _map, recall, precision, mrr, recall_cap, hole, mtebtask_results)
                }
    else:
        return ndcg, _map, recall, precision, mrr, recall_cap, hole, mtebtask_results

