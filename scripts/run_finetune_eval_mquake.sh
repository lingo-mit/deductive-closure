#!/bin/bash -l

DATAPATH="/projectnb/llamagrp/feyzanb/llama/dumped/mquake_single_edit"
SEED=0
lr=0.0005
epoch=30
TOTALTF=10
GRAPHTYPE="r|demi,demimh"
GRAPHSHORT="${GRAPHTYPE//|/_}"
OUTDIR="dumped/mquake_single_edit/llama-2-7b/n${TOTALTF}_${GRAPHSHORT}_${SEED}"
creak_dev="$DATAPATH/dev_n${TOTALTF}_${SEED}_pure.csv"
creak_test="$DATAPATH/dev_n${TOTALTF}_${SEED}_justq.csv"
creak_train="$OUTDIR/mquake_train_finetuning_questions.csv"
SUBDIR=$OUTDIR/epoch_${epoch}_lr_${lr}_questions
mkdir -p $SUBDIR

python finetune.py \
--creak_train $creak_train \
--creak_dev $creak_dev \
--creak_test $creak_test \
--lr $lr \
--epoch $epoch \
--eval_modes "test" \
--outdir $SUBDIR \
--max_new_tokens 64 \
--eval_method "mquake" \
--model_name_or_path "meta-llama/Llama-2-7b-hf" > $SUBDIR/finetune_log.txt 2>&1
