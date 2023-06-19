import logging
logger = logging.getLogger(__name__)
import copy
import torch.distributed as dist

from .util import cos_sim, dot_score
import torch
from typing import Dict, List


#Parent class for any dense model
class DenseRetrievalExactSearch:

    def __init__(self, model,
                 batch_size: int = 128,
                 query_batch_size: int = 128,
                 corpus_chunk_size: int = 20000,
                 return_cpu=True, add_qd_prompt=False, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.return_cpu = return_cpu
        self.add_qd_prompt = add_qd_prompt
        self.results = {}

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: List[int],
               score_function: str,
               return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            logger.info(f"Encoding Queries, #query={len(queries)}...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        if self.add_qd_prompt:
            queries = ['[Q]'+queries[qid] for qid in queries]
        else:
            queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
            return_cpu=self.return_cpu, convert_to_tensor=self.convert_to_tensor)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        self.results = {qid: {} for qid in query_ids}
        if self.add_qd_prompt:
            _corpus = []
            for cid in corpus_ids:
                d = copy.copy(corpus[cid])
                d['title'] = '[D]' + d['title']
                _corpus.append(d)
            corpus = _corpus
        else:
            corpus = [corpus[cid] for cid in corpus_ids]

        itr = range(0, len(corpus), self.corpus_chunk_size)
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            logger.info(f"Encoding Corpus in batches, #(doc)={len(corpus)}... \nWarning: This might take a while!")
            logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        for batch_num, corpus_start_idx in enumerate(itr):
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.info(f"Rank {dist.get_rank()}, Encoding Batch {batch_num+1}/{len(itr)}...")
            elif not dist.is_initialized():
                logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor=self.convert_to_tensor,
                return_cpu=self.return_cpu
                )

            # Compute similarites using either cosine-similarity or dot product
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                logger.info(f"Computing similarity, q={query_embeddings.shape}, d={sub_corpus_embeddings.shape}...")
            # single pass
            # cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            # cos_scores[torch.isnan(cos_scores)] = -1
            cos_scores = []
            num_q = len(query_embeddings)
            for i in range(num_q // self.query_batch_size + 1):
                # if dist.is_initialized() and dist.get_rank() == 0: logger.info(f"Query batch {i}...")
                q_emb = query_embeddings[i * self.query_batch_size: min((i + 1) * self.query_batch_size, num_q)]
                scores = self.score_functions[score_function](q_emb, sub_corpus_embeddings)
                cos_scores.append(scores)
            cos_scores = torch.cat(cos_scores)

            # Get top-k values, topk() is very slow on cpu
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                logger.info(f"Get top-k values...")
            cos_scores = cos_scores.float() if cos_scores.dtype == torch.float16 else cos_scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                logger.info(f"Merging results...")
                for query_itr in range(len(query_embeddings)):
                    # if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                    #     if query_itr % 100 == 0: print(query_itr)
                    query_id = query_ids[query_itr]
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                        if corpus_id != query_id:
                            self.results[query_id][corpus_id] = score
                    # print('before shortening', len(self.results[query_id]))
                    _results = {}
                    for cid, score in sorted(self.results[query_id].items(), key=lambda k: k[1], reverse=True)[:top_k]:
                        _results[cid] = score
                    self.results[query_id] = _results

                # print('after shortening', len(self.results[query_id]))
            # if dist.is_initialized() and dist.get_rank() == 0:
            #     logger.info(f"Rank {dist.get_rank()}, #score={len(self.results[query_id])}...")
            # elif not dist.is_initialized():
            #     logger.info(f'#score={len(self.results[query_id])}...')
        return self.results

