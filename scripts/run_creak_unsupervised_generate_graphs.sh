#!/bin/bash -l

# This script is used to generate graphs for the CREAK dataset in an unsupervised setting.
# More explanations of the parameters are available in run_generate_graphs_mquake.sh.

SEED=0
TOTALTF=10
GRAPHTYPE="be-l|i-l,c-l"
GRAPHSHORT="${GRAPHTYPE//|/_}"
HOME_FOLDER="/projectnb/llamagrp/feyzanb"
MODELNAME="$HOME_FOLDER/llama/llama-2-7b"
TOKENIZER="$HOME_FOLDER/llama/tokenizer.model"
OUTDIR="dumped/creak_unsupervised/llama-2-7b/n${TOTALTF}_${GRAPHSHORT}_${SEED}"
CACHE_HOME="cache/cache_seed${SEED}"
INPUT_PATH="$HOME_FOLDER/creak/data/creak/train.json"
INPUT_CLAIM_NAME_1="sentence"

mkdir -p $OUTDIR &&
mkdir -p $CACHE_HOME &&

torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 run_alt.py \
    --ckpt_dir $MODELNAME \
    --cache_path $CACHE_HOME/llama-2-7b_a_bunch.json \
    --tokenizer_path $TOKENIZER \
    --input_path $INPUT_PATH \
    --input_claim_name_1 $INPUT_CLAIM_NAME_1 \
    --query_num $TOTALTF \
    --output_path $OUTDIR \
    --short_gen_len 80 \
    --long_gen_len 348 \
    --temperature 0.9 \
    --drop_duplicates \
    --graph $GRAPHTYPE # > $OUTDIR/log.txt 2>&1