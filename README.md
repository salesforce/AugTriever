# AugTriever: Unsupervised Dense Retrieval by Scalable Data Augmentation

This repository contains the code and models of the paper "[AugTriever: Unsupervised Dense Retrieval by Scalable Data Augmentation](https://arxiv.org/abs/2212.08841)"

Our code is based on the following repositories:
- [SimCSE](https://github.com/princeton-nlp/SimCSE) released with the [SimCSE](https://arxiv.org/abs/2104.08821) paper.
- [Contriever](https://github.com/facebookresearch/contriever) released with the [Contriever](https://arxiv.org/abs/2109.07567) paper.
- [MoCo](https://github.com/facebookresearch/moco) released with the [MoCo](https://arxiv.org/pdf/1911.05722.pdf) paper.
- [DPR](https://github.com/facebookresearch/DPR) released with the [DPR](https://arxiv.org/abs/2004.04906) paper.
- [SPIDER](https://github.com/oriram/spider) released with the [SPIDER](https://arxiv.org/abs/1908.03847) paper.
- [MTEB](https://github.com/embeddings-benchmark/mteb) released with the [MTEB](https://arxiv.org/abs/2210.07316) paper.
- [BEIR](https://github.com/beir-cellar/beir) released with the [BEIR](https://arxiv.org/abs/2104.08663) paper.
- [SentEval](https://github.com/facebookresearch/SentEval) released with the [SentEval](https://arxiv.org/abs/1803.05449) paper.
- [E5](https://github.com/microsoft/unilm/tree/master/e5) released with the [E5](https://arxiv.org/pdf/2212.03533.pdf) paper.
- [rank_bm25](https://github.com/dorianbrown/rank_bm25)


### Data
[AugQ-Wiki](https://huggingface.co/datasets/memray/AugTriever-AugQ-Wiki) and [AugQ-CC](https://huggingface.co/datasets/memray/AugTriever-AugQ-CC) can be downloaded from Huggingface Hub.

### Checkpoints
Naming corresponds to Table 1 in the paper.

| Aug method    | Model    |  MM   |  BEIR (14 tasks)  |                           Download Link                                     |
|---------------|:---------|:-----:|:-----------------:|:---------------------------------------------------------------------------:|
| Hybrid-TQGen+ | MoCo     | 24.6  |       41.1        |  [[download](https://huggingface.co/memray/AugTriever-Hybrid-TQGen-plus)]   |
| Hybrid-All    | MoCo     | 23.5  |       39.4        |     [[download](https://huggingface.co/memray/AugTriever-Hybrid-All/)]      |
| Hybrid-TQGen  | MoCo     | 23.3  |       39.4        |  [[download](https://huggingface.co/memray/AugTriever-Hybrid-TQGen)]  |
| Doc-Title     | MoCo     | 21.8  |       38.7        |      [[download](https://huggingface.co/memray/AugTriever-DocTitle/)]       |
| QExt-PLM      | MoCo     | 20.6  |       38.2        |      [[download](https://huggingface.co/memray/AugTriever-QExt-PLM/)]       |
| TQGen-Topic   | MoCo     | 21.2  |       38.9        |     [[download](https://huggingface.co/memray/AugTriever-TQGen-Topic/)]     |
| TQGen-Title   | MoCo     | 21.8  |       39.3        |     [[download](https://huggingface.co/memray/AugTriever-TQGen-Title/)]     |
| TQGen-AbSum   | MoCo     | 23.2  |       39.6        |     [[download](https://huggingface.co/memray/AugTriever-TQGen-AbSum/)]     |
| TQGen-ExSum   | MoCo     | 23.0  |       39.4        |     [[download](https://huggingface.co/memray/AugTriever-TQGen-ExSum/)]     |
| TQGen-Topic   | InBatch  | 20.7  |       39.0        | [[download](https://huggingface.co/memray/AugTriever-TQGen-Topic-InBatch/)] |

### Run Training
A few scripts for starting training are placed in the folder `examples/traning`. For example:
```bash
cd $PATH_TO_REPO
sh examples/training/cc.moco.topic50.bs2048.gpu8.sh
```

### Run Evaluation
#### BEIR
Please refer to [BEIR](https://github.com/beir-cellar/beir) for data download.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python torch.distributed.launch --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=2255 eval_beir.py --model_name_or_path output_dir/augtriever-release/cc.T03b_title50.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5/ --dataset fiqa --metric dot --pooling average --per_gpu_batch_size 128 --beir_data_path data/beir/ --output_dir eval_dir/beir
```
#### ODQA
Please refer to [Spider](https://github.com/oriram/spider) for details about QA data download and processing. 
```bash
export EXP_DIR="output_dir/cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python torch.distributed.launch --nproc_per_node=8 --master_port=31133 --max_restarts=0 generate_passage_embeddings.py --model_name_or_path $EXP_DIR --output_dir $EXP_DIR/embeddings --passages data/nq/psgs_w100.tsv --per_gpu_batch_size 512
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_qa.py --model_name_or_path facebook/contriever --passages data/nq/psgs_w100.tsv --passages_embeddings "$EXP_DIR/embeddings/*" --qa_file data/nq/qas/*-test.csv,data/nq/qas/entityqs/test/P*.test.json --output_dir $EXP_DIR/qa_output --save_or_load_index
```

### Convert models
Convert to Huggingface BERT model
```bash
python convert_checkpoint_to_hf_bert.py --ckpt_path output_dir/cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5 --output_dir output_dir/cc.T03b_topic.inbatch.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5/hf_ckpt_bert --model_type shared
```
Convert to Huggingface DPR model
```bash
python convert_checkpoint_to_hf_dpr.py --ckpt_path output_dir/cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5 --output_dir output_dir/cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5/hf_ckpt_dpr --model_type shared
```

### Export scores
Replace the exp path in `gather_score_beir.py/gather_score_qa.py/gather_score_senteval.py` and run it. For example 
```bash
python gather_score_beir.py
```

## License

AugTriever is licensed under the [BSD 3-Clause License](LICENSE).

Evaluation codes that are forked from external repositories are placed in subfolders (e.g. `src/beir`,  `src/beireval`, `src/mteb`, `src/mtebeval`, `src/qa`, `src/senteval`). Please refer to LICENSE in each subfolder for their Copyright information.

## Citation

If you find the AugTriever code or models useful, please cite it by using the following BibTeX entry.

```BibTeX
@article{meng2022augtriever,
  title={AugTriever: Unsupervised Dense Retrieval by Scalable Data
Augmentation},
  author={Meng, Rui and Liu, Ye and Yavuz, Semih and Agarwal, Divyansh and Tu, Lifu and Yu, Ning and Zhang, Jianguo and Bhat, Meghana and Zhou, Yingbo},
  journal={arXiv preprint arXiv:2212.08841},
  year={2022}
}
```
