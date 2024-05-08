import numpy as np
from tqdm import tqdm
import os
import json
import ipdb
from dataclasses import fields
import argparse
from typing import Any
import itertools
import string


def parser_from_dataclass(data_class: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    for field in fields(data_class):
        kwargs = {
            "type": field.type,
            "default": field.default,
        }

        # If the field has no default value, it is considered required
        if field.default == field.default_factory:
            kwargs["required"] = True
            del kwargs["default"]

        parser.add_argument(f"--{field.name.replace('_', '-')}", **kwargs)

    return parser


def print_result(dialogs, results):
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


def make_list_prompt(sequences, num):
    samples = np.random.choice(sequences, num, replace=False)
    samples = [f"{i+1}. {s}" for i, s in enumerate(samples)]
    return "\n".join(samples)


def split_numbered_list(seq, num=3):
    try:
        return [seq.split(". ")[i] for i in range(1, num + 1)]
    except IndexError:
        return [""] * num


def split_numbered_single_line_list_gen(n=3, number_in_prompt=False):
    def split_numbered_single_line_list(seq):
        try:
            if number_in_prompt:
                return [seq.split(". ")[i].split("\n")[0] for i in range(0, n)]
            else:
                return [seq.split(". ")[i].split("\n")[0] for i in range(1, n + 1)]
        except IndexError:
            return [""] * n

    return split_numbered_single_line_list


def split_implication_of_related_claim_gen(n=3):
    def split_implication_of_related_claim(seq):
        try:
            if "Implications:" in seq:
                seq = seq.split("Implications:")[1]
            return split_numbered_single_line_list_gen(n=5)(seq)
            # ll = []
            # for line in seq.split("\n"):
            #     if "Implication " in line:
            #         candidate = line.split("Implication ")[1]
            #         if candidate[0].isdigit():
            #             candidate = candidate[2:].strip()
            #         ll.append(candidate)
            # return ll[:n]
        except IndexError:
            return [""] * n

    return split_implication_of_related_claim


def init_llama_dialog(system, query):
    dia = []
    dia.append({"role": "system", "content": system})
    dia.append({"role": "user", "content": query})
    return dia


def init_llama_input(inp, dialog=True):
    if dialog:
        return init_llama_dialog(*inp)
    else:
        return inp


def hash_dialog(dialog):
    return json.dumps(dialog)


def filter_condition(data, filter):
    if filter is None:
        return data
    colname, value = filter.split("=")
    return data.loc[data[colname] == value]


def util_parse_true_false(p: str):
    p = p.lower()
    if "verdict: true" in p:
        if "verdict: false" in p:
            # Return based on whichever comes first.
            if p.index("verdict: true") < p.index("verdict: false"):
                return 1
            else:
                return 0
        else:
            return 1
    elif "verdict: false" in p:
        return 0
    else:
        return -1


def util_contradictory(p: str):
    p = p.lower()
    if "not contradictory" in p:
        return 0
    elif "contradictory" in p:
        return 1
    else:
        return -1


def util_contradictory_(p: str):
    p = p.lower()
    if "verdict: contradictory" in p:
        if "verdict: not contradictory" in p:
            # Return based on whichever comes first.
            if p.index("verdict: contradictory") < p.index(
                "verdict: not contradictory"
            ):
                return 1
            else:
                return 0
        else:
            return 1
    elif "verdict: not contradictory" in p:
        return 0
    else:
        return -1


def util_dimplied(p: str):
    p = p.lower()
    if "does not imply" in p:
        return 0
    elif "implies" in p:
        return 1
    else:
        return -1


def util_dimplied_(p: str):
    p = p.lower()
    if "final verdict: does not imply" in p:
        if "final verdict: implies" in p:
            # Return based on whichever comes first.
            if p.index("final verdict: does not imply") < p.index(
                "final verdict: implies"
            ):
                return 0
            else:
                return 1
        else:
            return 0
    elif "final verdict: implies" in p:
        return 1
    else:
        return -1


def util_dimplied_v2(p: str):
    p = p.lower()
    if "entailment" in p:
        return 1
    else:
        return 0


def query_llama(inputs, opt, generator=None):
    if "chat" in opt.ckpt_dir:
        return query_llama_chat(inputs, opt, generator)
    else:
        return query_llama_text(inputs, opt, generator)


def query_llama_chat(dialogs, opt, generator=None):
    cache = Cache(opt.cache_path)

    if generator is None:
        from llama import Llama

        generator = Llama.build(
            ckpt_dir=opt.ckpt_dir,
            tokenizer_path=opt.tokenizer_path,
            max_seq_len=opt.max_seq_len,
            max_batch_size=opt.max_batch_size,
        )
    results = []
    tot = len(dialogs) // opt.max_batch_size
    for i in tqdm(range(0, len(dialogs), opt.max_batch_size), total=tot):
        batch = dialogs[i : i + opt.max_batch_size]
        if cache.check_cache(batch):
            results.extend(cache(batch))
            continue
        completions = generator.chat_completion(
            batch,
            max_gen_len=opt.max_gen_len,
            temperature=opt.temperature,
            top_p=opt.top_p,
        )
        results.extend(completions)
        cache.add(batch, completions)
    results = [r["generation"]["content"] for r in results]

    # del generator
    return results


def query_llama_text(texts, opt, generator=None, use_tqdm=True):
    cache = Cache(opt.cache_path)
    del_gen = False
    if generator is None:
        from llama import Llama

        generator = Llama.build(
            ckpt_dir=opt.ckpt_dir,
            tokenizer_path=opt.tokenizer_path,
            max_seq_len=opt.max_seq_len,
            # max_gen_len=opt.max_gen_len,
            max_batch_size=opt.max_batch_size,
        )
        del_gen = True
    results = []
    tot = len(texts) // opt.max_batch_size
    rng = range(0, len(texts), opt.max_batch_size)
    iterator = tqdm(rng, total=tot) if use_tqdm else rng
    for i in iterator:
        batch = texts[i : i + opt.max_batch_size]
        if cache.check_cache(batch):
            results.extend(cache(batch))
            continue
        completions = generator.text_completion(
            batch,
            max_gen_len=opt.max_gen_len,
            temperature=opt.temperature,
            top_p=opt.top_p,
        )
        results.extend(completions)
        cache.add(batch, completions)
    results = [r["generation"] for r in results]
    if del_gen:
        del generator

    return results


def identify_paths(graph):
    """
    Get each trace from the root to the leaf in the specified tree.
    e.g.
    graph: b-l|i-vi-l,c-vc-l|c-vc-l
    specification:
        1. generate a bunch of seed statements (b) and compute likelihoood (l)
        2.1. for each seed, generate a bunch of implications (i), double-check (vi) and compute likelihood (l)
        2.2. for each seed, generate a bunch of contradictions (c), double-check (vc) and compute likelihood (l)
        3. for each document from 2.1 and 2.2, generate contradictions, double-check (vc) and compute likelihood (l)
    branches: ["b-l|i-vi-l|c-vc-l", "b-l|c-vc-l|c-vc-l"]
    """
    # Split the input string into sets of nodes based on the '|' character
    node_sets = [nodes.split(",") for nodes in graph.split("|")]

    # Get all permutations of paths through the graph using itertools.product
    paths = list(itertools.product(*node_sets))

    return paths


def validate_graph(graph):
    assert True


def get_llama(opt):
    from llama import Llama

    generator = Llama.build(
        ckpt_dir=opt.ckpt_dir,
        tokenizer_path=opt.tokenizer_path,
        max_seq_len=opt.max_seq_len,
        max_batch_size=opt.max_batch_size,
    )
    return generator


def get_llama_likelihoods(texts, opt, generator=None):
    from llama import Llama

    cache = Cache(opt.cache_path)
    if generator is None:
        generator = Llama.build(
            ckpt_dir=opt.ckpt_dir,
            tokenizer_path=opt.tokenizer_path,
            max_seq_len=opt.max_seq_len,
            max_batch_size=opt.max_batch_size,
        )
    results = []

    tot = len(texts) // opt.max_batch_size
    for i in tqdm(range(0, len(texts), opt.max_batch_size), total=tot):
        batch = texts[i : i + opt.max_batch_size]
        if cache.check_cache(batch):
            results.extend(cache(batch))
            continue
        probs = generator.compute_likelihood(
            batch,
        )
        results.extend(probs)
        cache.add(batch, probs)

    # del generator
    return results


def eval_mquake(preds, labels):
    correct_l = []

    for pred, true in zip(preds, labels):
        correct = False

        if "Q:" in pred:
            pred = pred.split("Q:")[0]

        if type(true) is list:
            if any(
                [t.strip().lower() in pred.strip().lower() for t in true if len(t) > 2]
            ):
                correct = True
        elif type(true) is str:
            if "[" in true and "]" in true:
                true = eval(true)
                if any(
                    [
                        t.strip().lower() in pred.strip().lower()
                        for t in true
                        if len(t) > 2
                    ]
                ):
                    correct = True
            else:
                if pred.strip().lower() in true.strip().lower():
                    print(
                        "Warning checking pred in true, not vice versa (true, pred):",
                        true,
                        pred,
                    )
                    correct = True
        correct_l.append(True if correct else False)

    return correct_l


def eval_creak(preds, labels):
    correct_l = []

    for pred, true in zip(preds, labels):
        true = "".join([c for c in true if c not in string.punctuation])
        pred = "".join([c for c in pred if c not in string.punctuation])
        true = true.strip().lower()
        pred = pred.strip().lower()
        if true in pred:
            # Warning: can mark "True and False" as true.
            correct_l.append(True)
        else:
            correct_l.append(False)
    return correct_l


eval_perf = {
    "mquake": eval_mquake,
    "creak": eval_creak,
}


class Cache(object):
    def __init__(self, cache_path):
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            self.cache = json.load(open(cache_path, "r"))
        else:
            self.cache = {}

    def add(self, batch, batch_answers):
        if type(batch) == str:
            self.cache[batch] = batch_answers
        else:
            batch = [hash_dialog(d) for d in batch]
            for b, a in zip(batch, batch_answers):
                self.cache[b] = a
        json.dump(self.cache, open(self.cache_path, "w"))

    def check_cache(self, batch):
        if type(batch) == str:
            return batch in self.cache
        batch = [hash_dialog(d) for d in batch]
        return all([b in self.cache for b in batch])

    def __call__(self, batch):
        if type(batch) == str:
            return self.cache[batch]
        batch = [hash_dialog(d) for d in batch]
        return [self.cache[b] for b in batch]

    def __len__(self):
        return len(self.cache)
