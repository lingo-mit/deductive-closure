# How many entries to take as seeds.
TOTALTF=10

# Set seed.
SEED=0

# Project, model checkpoint, tokenizer folders.
HOME_FOLDER="/projectnb/llamagrp/feyzanb"
MODELNAME="$HOME_FOLDER/llama/llama-2-7b"
TOKENIZER="$HOME_FOLDER/llama/tokenizer.model"

# The kind of graph we would like to generate.
# Check out llama_query_util.py to see the available options and llama_prompts.py.
# e.g. demi: demographic implications, demimh: demographic multi-hop implications.
GRAPHTYPE="r|demi,demimh"

# Replace | with _ for file naming.
GRAPHSHORT="${GRAPHTYPE//|/_}"

# For artifacts e.g. where the output graph will be saved.
OUTDIR="dumped/mquake_single_edit/llama-2-7b/n${TOTALTF}_${GRAPHSHORT}_${SEED}"

# Cache to store model generations.
CACHE_HOME="cache/cache_mquake_seed${SEED}"

# Path to seed documents.
INPUT_PATH="$HOME_FOLDER/MQuAKE/datasets/MQuAKE-CF-3k-sing-edits_${TOTALTF}_${SEED}.json"

# The name of the field in the input json file that contains the seed documents.
INPUT_CLAIM_NAME_1="edit"

# Set skip best assignment as this is a supervised setting i.e. implications are automatically marked true.

mkdir -p $OUTDIR &&
mkdir -p $CACHE_HOME &&

torchrun --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 run_alt.py \
    --ckpt_dir $MODELNAME \
    --cache_path "${CACHE_HOME}/llama-2-7b-mquake.json" \
    --tokenizer_path $TOKENIZER \
    --input_path $INPUT_PATH \
    --input_claim_name_1 $INPUT_CLAIM_NAME_1 \
    --query_num $TOTALTF \
    --output_path $OUTDIR \
    --skip_best_assignment \
    --long_gen_len 320 \
    --graph $GRAPHTYPE