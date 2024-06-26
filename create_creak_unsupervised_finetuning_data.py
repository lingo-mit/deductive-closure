import pandas as pd
import re
import random

# Create fine-tuning dataset for ordinary llama.
input_prompt = """Label the following fact as true or false. Answer only with True or False.
Fact: {user_msg}
Label:"""

# Set seed
random.seed(42)

modelname = "llama-2-7b"


def clean_statement(text):
    # if the text contains the more than 10 repetitive characters, return None.
    if type(text) is not str or re.search(r"(.)\1{10,}", text):
        return None
    else:
        # Remove surrounding " and '.
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        return text


for num in [10]:
    for path in ["be-l_i-l,c-l"]:
        for seed in [0]:
            creak_train = pd.read_csv(
                f"dumped/creak_unsupervised/{modelname}/n{num}_{path}_{seed}/all_nodes.csv"
            )
            creak_train = creak_train[["name", "truth_value"]]
            # Clean for repetitive chars.
            creak_train["name"] = creak_train["name"].apply(clean_statement)
            # Drop empty.
            creak_train = creak_train.dropna()
            # Labels
            creak_train["label"] = creak_train["truth_value"].apply(
                lambda x: "True." if x else "False."
            )
            creak_train["input_prompt"] = creak_train.apply(
                lambda x: input_prompt.format(user_msg=x["name"]), axis=1
            )
            creak_train = creak_train[["input_prompt", "label"]]
            # Shuffle.
            creak_train = creak_train.sample(frac=1, random_state=42)
            # Select unique prompts
            creak_train = creak_train.drop_duplicates(subset=["input_prompt"])
            creak_train.to_csv(
                f"dumped/creak_unsupervised/{modelname}/n{num}_{path}_{seed}/creak_train_finetuning.csv",
                index=False,
            )
