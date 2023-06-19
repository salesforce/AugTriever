# https://github.com/shmsw25/GraphRetriever

#!/usr/bin/env python
import json
import math
import os
import random
import string
from collections import defaultdict

import re
import numpy as np
from multiprocessing import Pool, cpu_count

import tqdm

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


def calc_tf(tokens):
    tf = defaultdict(int)
    for word in tokens:
        tf[word] += 1
    return tf


def default_tokenizer_fn(text):
    text = re.sub('[' + string.punctuation + ']', ' ', text).lower().split()
    return text


class BM25:
    def __init__(self, corpus=None, tokenizer=None, num_thread=cpu_count()):
        self.avg_doc_len = 0
        self.df = {}
        self.idf = {}
        self.doc_tfs = []  # term frequency of each doc
        self.doc_lens = []  # number of tokens of each doc
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = default_tokenizer_fn
        if corpus:
            self.corpus_size = len(corpus)
            print('Start tokenizing corpus')
            corpus = self._tokenize_corpus(corpus, num_thread)
            print('Start computing DF')
            self._calc_df(corpus)
            print('Start computing IDF')
            self._calc_idf(self.df)
            print('Done initializing BM25 model')
            self.corpus = corpus

    def _calc_df(self, corpus):
        df_corpus = defaultdict(int)  # word -> number of documents with word
        num_word = 0
        for doc in tqdm.tqdm(corpus, 'Computing DF'):
            self.doc_lens.append(len(doc))
            num_word += len(doc)
            tf_doc = defaultdict(int)
            for word in doc:
                if word not in tf_doc:
                    df_corpus[word] += 1
                tf_doc[word] += 1
            self.doc_tfs.append(tf_doc)
        self.avg_doc_len = num_word / self.corpus_size
        self.df = df_corpus
        print(f'Corpus processing completed. #(doc)={self.corpus_size}, #(word)={num_word}, avg(doc_len)={self.avg_doc_len}, #(vocab)={len(df_corpus)}')

    def _tokenize_corpus(self, corpus, num_thread):
        pool = Pool(num_thread)
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def save_to_json(self, json_path):
        print('Saving BM25 model to', json_path)
        save_items = self.__dict__
        # only save statistics
        del save_items['doc_tfs']
        del save_items['doc_lens']
        del save_items['tokenizer']
        del save_items['corpus']
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as jf:
            jf.write(json.dumps(save_items))
        print('Saving completed')

    def load_from_json(self, json_path):
        print('Loading BM25 model from', json_path)
        self_dict = json.load(open(json_path, 'r'))
        for k, v in self_dict.items():
            setattr(self, k, v)
        print('Loading completed')


class BM25Okapi(BM25):
    def __init__(self, corpus=None, tokenizer=None, k1=1.2, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, df_dict):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in df_dict.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_lens)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_tfs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)))
        return score

    def get_score_for_chunks(self, doc, chunks):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        doc_tf = calc_tf(self.tokenizer(doc))
        scores = [0.0] * len(chunks)
        for qid, chunk in enumerate(chunks):
            chunk_tokens = self.tokenizer(chunk)
            for w in chunk_tokens:
                tf = doc_tf.get(w, 0)
                scores[qid] += (self.idf.get(w) or 0) * (tf * (self.k1 + 1) /
                                                   (tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_doc_len)))
            scores[qid]
        return scores

    def batch_rank_chunks(self, batch_docs, batch_chunks):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        chunk_idx = []
        for doc, chunks in zip(batch_docs, batch_chunks):
            scores = self.get_score_for_chunks(doc, chunks)
            chunk_idx.append(np.argmax(scores))

            # chunk_score_pairs = sorted(zip(range(len(chunks)), chunks, scores), key=lambda t: t[2], reverse=True)
            # print('*' * 30)
            # print(doc)
            # for chunk in chunk_score_pairs:
            #     print(f'[{chunk[0]}](score={chunk[2]}) {chunk[1]}')
            # print()
        return chunk_idx


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.2, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_lens)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_tfs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_tfs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_lens)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_tfs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.2, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_lens)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_tfs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_tfs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_lens)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_tfs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len) + q_freq))
        return score.tolist()
