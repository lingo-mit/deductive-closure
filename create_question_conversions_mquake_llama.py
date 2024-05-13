"""
Use this script to prompt LLaMa to convert statements into question format.
The statements themselves will be used as answers.
"""

from util import query_llama_text
from dataclasses import dataclass
import pandas as pd
import os
from tqdm import tqdm
from llama import Llama


# Edit the directories for llama and tokenizer.
@dataclass
class opt:
    ckpt_dir: str = "/projectnb/llamagrp/feyzanb/llama/llama-2-7b"
    tokenizer_path: str = "/projectnb/llamagrp/feyzanb/llama/tokenizer.model"
    max_batch_size: int = 12
    max_gen_len: int = 64
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 1024
    cache_path: str = "cache/cache_mquake_question_conversion.json"


# Few-shot prompt for question conversion.
prompt_template = """Sentence: Kate Winslet is a citizen of the UK.
Question: Which country is Kate Winslet a citizen of?
Sentence: Ukraine is a country in Europe.
Question: Which continent is Ukraine in?
Sentence: The country where Priyanka Chopra is from is India. The capital of India is New Delhi.
Question: What is the capital of the country where Priyanka Chopra is from?
Sentence: {sentence}
Question:"""

# Create llama.
generator = Llama.build(
    ckpt_dir=opt.ckpt_dir,
    tokenizer_path=opt.tokenizer_path,
    max_seq_len=opt.max_seq_len,
    max_batch_size=opt.max_batch_size,
)

# Iterate over the generated graphs to convert statements into questions.
totaltf = [10, 20, 50, 100, 1000]
seeds = [0]
graphs = ["r_demi,demimh"]

for tf in tqdm(totaltf, desc="totaltf"):
    for seed in seeds:
        for graph in graphs:
            basepath = f"dumped/mquake_single_edit/llama-2-7b/n{tf}_{graph}_{seed}"
            path = os.path.join(basepath, "all_nodes.csv")
            try:
                sentences = pd.read_csv(path)["name"]
            except FileNotFoundError:
                print(f"File not found for {path}")
                continue

            texts = [
                prompt_template.format(sentence=sentence) for sentence in sentences
            ]
            results = query_llama_text(texts, opt, generator)
            results = [r.split("Sentence:")[0].strip() for r in results]

            df = pd.DataFrame({"input_prompt": results, "label": sentences})

            # Drop if input_prompt or label is empty.
            df = df.dropna(subset=["input_prompt", "label"])
            df.to_csv(f"{basepath}/mquake_train_finetuning_questions.csv", index=False)
