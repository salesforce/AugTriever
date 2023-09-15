#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_RANK=0
export WORLD_SIZE=8

export TOKENIZERS_PARALLELISM=false
export NUM_WORKER=2
export MAX_STEPS=100000

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export EXP_NAME=cc.RC50-DocTitle50.moco-2e14.contriever256.bert-base-uncased.avg.dot.q128d256.step100k.bs2048.lr5e5
export PROJECT_DIR=output_dir/$EXP_NAME
mkdir -p $PROJECT_DIR
cp "$0" $PROJECT_DIR  # copy bash to project dir
echo $PROJECT_DIR

export WANDB_NAME=$EXP_NAME
export WANDB_API_KEY=$YOUR_WANDB_API_KEY
export WANDB_PROJECT=unsup_retrieval_cc
export WANDB_DIR=$PROJECT_DIR
mkdir -p $WANDB_DIR/wandb

nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=31133 --max_restarts=0 train.py --model_name_or_path bert-base-uncased --arch_type moco --train_file data/cc/pilecc_uqg.jsonl --data_type hf --data_pipeline_name  contriever256 --pseudo_query_names "{'random-crop':0.5,'title':0.5}" --remove_unused_columns False --sim_type dot --queue_size 16384 --momentum 0.9995 --output_dir $PROJECT_DIR --cache_dir /export/home/data/pretrain/.cache --max_steps $MAX_STEPS --warmup_steps 10000 --logging_steps 100 --eval_steps 10000 --save_steps 1000000 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --dataloader_num_workers $NUM_WORKER --learning_rate 5e-5 --max_q_tokens 128 --max_d_tokens 256 --evaluation_strategy steps --load_best_model_at_end --overwrite_output_dir --do_train --do_eval --run_name $EXP_NAME --fp16 --seed 42 --report_to wandb --wiki_passage_path data/qa/nq/psgs_w100.tsv --qa_datasets_path data/qa/nq/qas/*-test.csv,data/qa/nq/qas/entityqs/test/P*.test.json > $PROJECT_DIR/nohup.log 2>&1 & echo $! > run.pid
