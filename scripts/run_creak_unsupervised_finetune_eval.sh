#!/bin/bash -l

DATAPATH="/projectnb/llamagrp/feyzanb/deductive-closure"

cnt=0
lr=0.0001
epoch=30
SEED=0
TOTALTF=10
MODELNAME="Llama-2-7b-hf"
GRAPHSHORT="be-l_i-l,c-l"
OUTDIR="$DATAPATH/dumped/creak_unsupervised/llama-2-7b/n${TOTALTF}_${GRAPHSHORT}_${SEED}"
creak_dev="/projectnb/llamagrp/feyzanb/llama/data/creak/creak_n200_val/creak_val_from_train_200tf_ordinary_instruction.csv"
creak_test="/projectnb/llamagrp/feyzanb/llama/data/creak/creak_dev_ordinary_instruction.csv"
creak_train="$OUTDIR/creak_train_finetuning.csv"
SUBDIR=$OUTDIR/epoch_${epoch}_lr_${lr}


mkdir -p $SUBDIR

python finetune.py \
--creak_train $creak_train \
--creak_dev $creak_dev \
--creak_test $creak_test \
--lr $lr \
--eval_method creak \
--epoch $epoch \
--batch_size 4 \
--eval_modes "dev" \
--model_name_or_path meta-llama/$MODELNAME \
--outdir $SUBDIR # > $SUBDIR/log.txt 2>&1