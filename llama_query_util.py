import argparse
import pandas as pd
import os
import ipdb
from dataclasses import dataclass
from llama_prompts import *

# from llama_prompts import (
#     get_similar_contradicting_claim_prompt,
#     get_similar_contradicting_v2_claim_prompt,
#     get_similar_contradicting_v3_claim_prompt,
#     get_verification_prompt,
#     get_decide_contradiction_prompt,
#     get_claim_rephrasing_prompt,
#     get_claim_implications_prompt,
#     get_claim_family_implications_prompt,
#     get_claim_family_v2_implications_prompt,
#     get_decide_implication_prompt,
#     get_decide_implication_v2_prompt,
#     get_a_bunch_of_claims_prompt,
#     get_claim_truth_value_prompt,
#     get_a_bunch_of_claims_wexample_prompt,
#     get_claim_demographic_implications_prompt,
#     get_claim_demographic_implications_prompt_multihop,
#     get_similar_claims_prompt,
#     get_claim_implications_v2_prompt,
#     get_claim_implications_v3_prompt,
#     get_similar_contradicting_claim_prompt_chat,
#     get_similar_contradicting_v2_claim_prompt_chat,
#     get_verification_prompt_chat,
#     get_decide_contradiction_prompt_chat,
#     get_claim_rephrasing_prompt_chat,
#     get_claim_implications_prompt_chat,
#     get_claim_family_implications_prompt_chat,
#     get_claim_family_v2_implications_prompt_chat,
#     get_decide_implication_prompt_chat,
#     get_decide_implication_v2_prompt_chat,
#     get_a_bunch_of_claims_prompt_chat,
#     get_claim_truth_value_prompt_chat,
#     get_a_bunch_of_claims_wexample_prompt_chat,
#     get_claim_demographic_implications_prompt_chat,
#     get_claim_demographic_implications_prompt_multihop_chat,
#     get_similar_claims_prompt_chat,
# )
from util import (
    init_llama_input,
    query_llama,
    get_llama_likelihoods,
    split_numbered_single_line_list_gen,
    filter_condition,
    util_parse_true_false,
    util_contradictory,
    util_dimplied,
    util_contradictory_,
    util_dimplied_,
    util_dimplied_v2,
    parser_from_dataclass,
    split_implication_of_related_claim_gen,
)


# @dataclass
# class Args:
#     ckpt_dir: str
#     tokenizer_path: str
#     cache_path: str
#     input_path: str
#     output_path: str = "claims.json"
#     input_claim_name_1: str = "claim1"
#     input_claim_name_2: str = "claim2"
#     filter: str = None
#     operation: str = "similar"
#     temperature: float = 0.6
#     top_p: float = 0.9
#     max_seq_len: int = 1024
#     max_batch_size: int = 12
#     max_gen_len: int = 512
#     short_gen_len: int = 128
#     long_gen_len: int = 512
#     query_num: int = 999999
#     add_original: bool = False
#     drop_duplicates: bool = False


def data_read_util(path, query_num, filter):
    print("Reading data from", path)
    data = pd.read_json(path, lines=True)
    query_num = min(query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, filter)
    return data


def generate_more(opt, generator=None):
    """
    This procedure adds another level to the tree of claims.
    It is used to sample seeds claims, similar claims, implications
    and contradictions.
    """
    data = data_read_util(opt.input_path, opt.query_num, opt.filter)
    claims = data[opt.input_claim_name_1]
    promptmap = PROMPTMAP_CHAT if "chat" in opt.ckpt_dir else PROMPTMAP
    promptmake = promptmap[opt.operation]
    prompts = [promptmake(c) for c in claims]
    inps = [init_llama_input(sq, "chat" in opt.ckpt_dir) for sq in prompts]
    optmaxprev = opt.max_gen_len
    opt.max_gen_len = opt.long_gen_len
    results = query_llama(inps, opt, generator)
    opt.max_gen_len = optmaxprev
    colname = f"{opt.input1nick}.{opt.opnick}"
    data[f"{colname}_raw"] = results
    postprocessmap = POSTPROCESSMAP_CHAT if "chat" in opt.ckpt_dir else POSTPROCESSMAP
    postf = postprocessmap[opt.operation]
    results = [postf(r) for r in results]
    # if opt.add_original:
    #     results = [c + [o] for c, o in zip(results, claims)]
    data[f"{colname}"] = results
    data = data.explode(f"{colname}")
    data.to_json(opt.output_path, orient="records", lines=True)
    data = data[[opt.input_claim_name_1, colname]]
    data.columns = ["key", "value"]
    return colname, data["key"].tolist(), data["value"].tolist()


def verify(opt, generator=None):
    """
    This procedure is used to verify a claim e.g. whether it is true.
    """
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    claims = data[opt.input_claim_name_1]
    promptmap = PROMPTMAP_CHAT if "chat" in opt.ckpt_dir else PROMPTMAP
    promptmake = promptmap[opt.operation]
    prompts = [promptmake(c) for c in claims]
    optmaxprev = opt.max_gen_len
    opt.max_gen_len = opt.short_gen_len
    inps = [init_llama_input(sq, "chat" in opt.ckpt_dir) for sq in prompts]
    results = query_llama(inps, opt, generator)
    opt.max_gen_len = optmaxprev
    colname = f"{opt.input1nick}.{opt.opnick}"
    data[f"{colname}_raw"] = results
    postprocessmap = POSTPROCESSMAP_CHAT if "chat" in opt.ckpt_dir else POSTPROCESSMAP
    postf = postprocessmap[opt.operation]
    results = [postf(r) for r in results]
    data[f"{colname}"] = results
    data.to_json(opt.output_path, orient="records", lines=True)
    return colname


def verify_pair(opt, generator=None):
    """
    This procedure is used to verify a pair of claims e.g.
    whether they are contradictory.
    """
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    claims1 = data[opt.input_claim_name_1]
    claims2 = data[opt.input_claim_name_2]
    promptmap = PROMPTMAP_CHAT if "chat" in opt.ckpt_dir else PROMPTMAP
    promptmake = promptmap[opt.operation]
    prompts = [promptmake(c1, c2) for c1, c2 in zip(claims1, claims2)]
    inps = [init_llama_input(sq, "chat" in opt.ckpt_dir) for sq in prompts]
    optmaxprev = opt.max_gen_len
    opt.max_gen_len = opt.short_gen_len
    results = query_llama(inps, opt, generator)
    opt.max_gen_len = optmaxprev
    colpair = opt.input1nick + "." + opt.input2nick + "." + opt.opnick
    data[colpair + "_raw"] = results
    postprocessmap = POSTPROCESSMAP_CHAT if "chat" in opt.ckpt_dir else POSTPROCESSMAP
    postf = postprocessmap[opt.operation]
    results = [postf(r) for r in results]
    data[colpair] = results
    data.to_json(opt.output_path, orient="records", lines=True)
    return (
        results,
        data[opt.input_claim_name_1].tolist(),
        data[opt.input_claim_name_2].tolist(),
    )


def likelihood(opt, generator=None):
    """
    Compute the probability of a sequence. It is used to compute
    the probability of a claim being true.
    """
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    texts = data[opt.input_claim_name_1]
    promptmap = PROMPTMAP_CHAT if "chat" in opt.ckpt_dir else PROMPTMAP
    promptmake = promptmap[opt.operation]
    prompts = [promptmake(c) for c in texts]
    results = get_llama_likelihoods(prompts, opt, generator)
    colname = opt.input1nick + "." + opt.opnick
    data[colname + "_raw"] = results
    postprocessmap = POSTPROCESSMAP_CHAT if "chat" in opt.ckpt_dir else POSTPROCESSMAP
    postf = postprocessmap[opt.operation]
    results = [postf(r) for r in results]
    data[colname] = results
    data.to_json(opt.output_path, orient="records", lines=True)
    return results, data[opt.input_claim_name_1].tolist()


def query_main(opt, generator=None):
    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)
    opt.opnick = OPERATIONNICKMAP[opt.operation]
    opt.input1nick = opt.input_claim_name_1  # [:2]
    opt.input2nick = opt.input_claim_name_2  # [:2]

    return OPERATIONMAP[opt.operation](opt, generator)


# def set_defaults(opt):
#     args = Args(
#         ckpt_dir=opt.ckpt_dir,
#         tokenizer_path=opt.tokenizer_path,
#         cache_path=opt.cache_path,
#         input_path=opt.input_path,
#         output_path=opt.output_path,
#         input_claim_name_1=opt.input_claim_name_1,
#         input_claim_name_2=opt.input_claim_name_2,
#         filter=opt.filter,
#         temperature=opt.temperature,
#         top_p=opt.top_p,
#         max_seq_len=opt.max_seq_len,
#         max_batch_size=opt.max_batch_size,
#         max_gen_len=opt.max_gen_len,
#         query_num=opt.query_num,
#         short_gen_len=opt.short_gen_len,
#         long_gen_len=opt.long_gen_len,
#         add_original=opt.add_original,
#     )

#     return args


PROMPTMAP = {
    "similar_contradicting": get_similar_contradicting_claim_prompt,
    "similar_contradicting_v2": get_similar_contradicting_v2_claim_prompt,
    "similar_contradicting_v3": get_similar_contradicting_v3_claim_prompt,
    "verification": get_verification_prompt,
    "contradiction": get_decide_contradiction_prompt,
    "rephrase": get_claim_rephrasing_prompt,
    "implication": get_claim_implications_prompt,
    "implication_v2": get_claim_implications_v2_prompt,
    "implication_v3": get_claim_implications_v3_prompt,
    "fam_implication": get_claim_family_implications_prompt,
    "fam_implication_v2": get_claim_family_v2_implications_prompt,
    "dem_implication": get_claim_demographic_implications_prompt,
    "implied": get_decide_implication_prompt,
    "implied_v2": get_decide_implication_v2_prompt,
    "bunch_of_claims": get_a_bunch_of_claims_prompt,
    "likelihood": get_claim_truth_value_prompt,
    "bunch_of_claims_wexample": get_a_bunch_of_claims_wexample_prompt,
    "dem_mh_implication": get_claim_demographic_implications_prompt_multihop,
    "transductive_gen": get_similar_claims_prompt,
}

PROMPTMAP_CHAT = {
    "similar_contradicting": get_similar_contradicting_claim_prompt_chat,
    "similar_contradicting_v2": get_similar_contradicting_v2_claim_prompt_chat,
    "verification": get_verification_prompt_chat,
    "contradiction": get_decide_contradiction_prompt_chat,
    "rephrase": get_claim_rephrasing_prompt_chat,
    "implication": get_claim_implications_prompt_chat,
    "fam_implication": get_claim_family_implications_prompt_chat,
    "fam_implication_v2": get_claim_family_v2_implications_prompt_chat,
    "dem_implication_v2": get_claim_demographic_implications_prompt_chat,
    "implied": get_decide_implication_prompt_chat,
    "implied_v2": get_decide_implication_v2_prompt_chat,
    "bunch_of_claims": get_a_bunch_of_claims_prompt_chat,
    "likelihood": get_claim_truth_value_prompt_chat,
    "bunch_of_claims_wexample": get_a_bunch_of_claims_wexample_prompt_chat,
    "dem_mh_implication": get_claim_demographic_implications_prompt_multihop_chat,
    "transductive_gen": get_similar_claims_prompt_chat,
}

PREPROCESSMAP = {
    "similar_contradicting": lambda x: x,
    "similar_contradicting_v2": lambda x: x,
    "similar_contradicting_v3": lambda x: x,
    "verification": lambda x: x,
    "contradiction": lambda x: x,
    "rephrase": lambda x: x,
    "implication": lambda x: x,
    "implication_v2": lambda x: x,
    "implication_v3": lambda x: x,
    "fam_implication": lambda x: x,
    "fam_implication_v2": lambda x: x,
    "dem_implication": lambda x: x,
    "implied": lambda x: x,
    "implied_v2": lambda x: x,
    "bunch_of_claims": lambda x: x,
    "likelihood": lambda x: x,
    "bunch_of_claims_wexample": lambda x: x,
    "dem_mh_implication": lambda x: x,
    "transductive_gen": lambda x: x,
}


POSTPROCESSMAP = {
    "similar_contradicting": split_numbered_single_line_list_gen(n=3),
    "similar_contradicting_v2": split_numbered_single_line_list_gen(n=3),
    "similar_contradicting_v3": split_numbered_single_line_list_gen(n=3),
    "verification": util_parse_true_false,
    "contradiction": util_contradictory_,
    "rephrase": split_numbered_single_line_list_gen(n=3),
    "implication": split_numbered_single_line_list_gen(n=3),
    "implication_v2": split_numbered_single_line_list_gen(n=3),
    "implication_v3": split_numbered_single_line_list_gen(n=3),
    "fam_implication": split_numbered_single_line_list_gen(n=3),
    "fam_implication_v2": split_numbered_single_line_list_gen(n=3),
    "dem_implication": split_numbered_single_line_list_gen(n=5),
    "dem_mh_implication": split_implication_of_related_claim_gen(n=5),
    "implied": util_dimplied_,
    "implied_v2": util_dimplied_v2,
    "bunch_of_claims": split_numbered_single_line_list_gen(n=10, number_in_prompt=True),
    "likelihood": lambda x: round(x[0], 3),
    "bunch_of_claims_wexample": split_numbered_single_line_list_gen(
        n=10, number_in_prompt=True
    ),
    "transductive_gen": split_numbered_single_line_list_gen(n=5),
}

POSTPROCESSMAP_CHAT = {
    "similar_contradicting": split_numbered_single_line_list_gen(n=3),
    "similar_contradicting_v2": split_numbered_single_line_list_gen(n=3),
    "similar_contradicting_v3": split_numbered_single_line_list_gen(n=3),
    "verification": util_parse_true_false,
    "contradiction": util_contradictory,
    "rephrase": split_numbered_single_line_list_gen(n=3),
    "implication": split_numbered_single_line_list_gen(n=3),
    "fam_implication": split_numbered_single_line_list_gen(n=3),
    "fam_implication_v2": split_numbered_single_line_list_gen(n=3),
    "dem_implication": split_numbered_single_line_list_gen(n=5),
    "dem_mh_implication": split_implication_of_related_claim_gen(n=5),
    "implied": util_dimplied,
    "implied_v2": util_dimplied_v2,
    "bunch_of_claims": split_numbered_single_line_list_gen(n=10),
    "likelihood": lambda x: round(x[0], 3),
    "bunch_of_claims_wexample": split_numbered_single_line_list_gen(n=10),
    "transductive_gen": split_numbered_single_line_list_gen(n=5),
}

OPERATIONMAP = {
    "similar_contradicting": generate_more,
    "similar_contradicting_v2": generate_more,
    "similar_contradicting_v3": generate_more,
    "verification": verify,
    "contradiction": verify_pair,
    "rephrase": generate_more,
    "implication": generate_more,
    "implication_v2": generate_more,
    "implication_v3": generate_more,
    "fam_implication": generate_more,
    "fam_implication_v2": generate_more,
    "dem_implication": generate_more,
    "dem_mh_implication": generate_more,
    "implied": verify_pair,
    "implied_v2": verify_pair,
    "bunch_of_claims": generate_more,
    "likelihood": likelihood,
    "bunch_of_claims_wexample": generate_more,
    "transductive_gen": generate_more,
}

OPERATIONNICKMAP = {
    "similar_contradicting": "sc",
    "similar_contradicting_v2": "sc",
    "similar_contradicting_v3": "sc",
    "verification": "vr",
    "contradiction": "cv",
    "rephrase": "re",
    "implication": "im",
    "implication_v2": "im",
    "implication_v3": "im",
    "fam_implication": "fi",
    "fam_implication_v2": "fi",
    "dem_implication": "demi",
    "dem_mh_implication": "demimh",
    "implied": "iv",
    "implied_v2": "iv",
    "bunch_of_claims": "bc",
    "likelihood": "li",
    "bunch_of_claims_wexample": "bce",
    "transductive_gen": "tg",
}

NICK2OPERATION = {
    "i": "implication",
    "fi": "fam_implication",
    "demi": "dem_implication",
    "demimh": "dem_mh_implication",
}

CACHESUF = {
    "i": "_genimpl",
    "fi": "_genfamimpl",
    "demi": "_gendemimpl",
    "demimh": "_gendemimhimpl",
}
