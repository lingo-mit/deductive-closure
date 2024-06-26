# Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability

Paper: https://arxiv.org/abs/2401.08574
Project page: https://lingo-mit.github.io/deductive-closure/

Afra Feyza Akyürek, Ekin Akyürek, Leshem Choshen, Derry Wijaya, and Jacob Andreas

![Figure 1: Overview of Deductive Closure Training (DCT). To improve coherence of language model predictions, we begin with a set of seed documents (pink-highlighted), then use an LM to generate a collection of statements implied by or contradicting these. Next, we find the most probable subset before fine-tuning the LM on the selected subset.](teaser_small.jpg)

## Abstract

While language models (LMs) can sometimes generate factually correct text and estimate truth values of individual claims, these generally do not reflect a globally coherent, manipulable model of the world. As a consequence, current LMs also generate incorrect or nonsensical content, and are difficult to edit and bring up to date. We present a method called Deductive Closure Training (DCT) that uses LMs themselves to identify implications of (and contradictions within) the text that they generate, yielding an efficient self-supervised procedure for improving LM factuality. Given a collection of seed documents, DCT prompts LMs to generate additional text implied by these documents, reason globally about the correctness of this generated text, and finally fine-tune on text inferred to be correct. Given seed documents from a trusted source, DCT provides a tool for supervised model updating; if seed documents are sampled from the LM itself, DCT enables fully unsupervised fine-tuning for improved coherence and accuracy. Across the [CREAK](https://arxiv.org/abs/2109.01653), [MQUaKE](https://arxiv.org/abs/2305.14795), and [Reversal Curse](https://arxiv.org/abs/2309.12288) datasets, supervised DCT improves LM fact verification and text generation accuracy by 3-26%; on CREAK fully unsupervised DCT improves verification accuracy by 12%. These results show that LMs' reasoning capabilities during inference can be leveraged during training to improve their reliability.

## Running Experiments

Create an environment and install the required packages in `requirements.txt`. Data is available in [Google Drive](https://drive.google.com/drive/folders/1Qac61vX36PZwn7rgZPNAwV8T2cDTYqei?usp=sharing).

### 1. Generating DCT graphs

Checkout the sample scripts under `scripts` e.g. use `sh scripts/run_generate_graphs_mquake.sh`. Graphs will appear under `dumped` directory. For Unsupervised DCT for CREAK, check out `scripts/run_creak_unsupervised_generate_graphs.sh`.

### 2. Converting Statements into Evaluation Format (MQUaKE, Reversal Curse)

Checkout the sample scripts under `scripts` e.g. use `sh scripts/run_question_conversion_mquake.sh`. This script calls `create_question_conversions_mquake_llama.py`, make sure to point to the correct llama checkpoint and tokenizer paths. Within this script you can specify the paths to the graphs for which you want to convert statements to questions. The converted files will appear under the same directories as the graphs. For Unsupervised DCT for CREAK, check out `scripts/run_creak_unsupervised_generate_finetuning.sh`.

### 3. Fine-Tuning and Evaluation

Use `sh scripts/run_finetune_eval_mquake.sh` for fine-tuning LLaMa with peft library. This script requires the huggingface version of llama. If you don't have it [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) is a conversion script or just let the script download the model for you. The adapter checkpoints and predictions for test set will appear under the same directories as the graphs. For test examples, see the Google Drive link above. For Unsupervised DCT, checkout `scripts/scripts/run_creak_unsupervised_finetune_eval.sh`.
