import argparse
from llama_query_util import (
    query_main,
    data_read_util,
    CACHESUF,
    NICK2OPERATION,
)
from util import get_llama, identify_paths, validate_graph
import ipdb
from graph_manager import GraphManager
import os


def runb(args, generator=None):
    """
    Self-generate seed documents.
    """
    # args.input_path = args.entry_point
    # args.input_claim_name_1 = args.init_input_claim_name_1
    # args.query_num = args.init_query_num
    args.cache_path = args.cache_path.split(".json")[0] + "_a_bunch.json"
    args.output_path = args.output_path + "/bunch_of_claims.json"
    args.operation = "bunch_of_claims"
    args.query_num = 1
    colname, _, gens = query_main(args, generator)
    args.query_num = 9999999
    return colname, gens


def runbe(args, generator=None):
    """
    Self-generate seed documents while using an arbitrary example
    in the prompt to promote diversity.
    """
    # args.input_path = args.entry_point
    # args.input_claim_name_1 = args.init_input_claim_name_1
    # args.query_num = args.init_query_num
    args.cache_path = args.cache_path.split(".json")[0] + "_a_bunch.json"
    args.output_path = args.output_path + "/bunch_of_claims_wexample.json"
    args.operation = "bunch_of_claims_wexample"
    colname, _, gens = query_main(args, generator)
    args.query_num = 9999999
    return colname, gens


def runtg(args, generator=None, extend=False):
    """
    Seed documents are generated from the transductive setting.
    If extend is True, then the seed documents are extended to
    include the original (dev) documents. Otherwise, seed documents
    are only those similar (prompt p_rel in the paper) to the
    dev documents.
    """
    # args.input_path = args.entry_point
    # args.input_claim_name_1 = args.init_input_claim_name_1
    # args.query_num = args.init_query_num
    args.cache_path = args.cache_path.split(".json")[0] + "_transductive.json"
    args.output_path = args.output_path + "/transductive_gen.json"
    args.operation = "transductive_gen"
    add_original_ = args.add_original
    args.add_original = extend
    colname, _, gens = query_main(args, generator)
    args.add_original = add_original_
    args.query_num = 9999999
    return colname, gens


def runr(args, likelihood_colname):
    """
    Read seed documents externally, should specify probability of
    being true in the likelihood_colname column.
    """
    # args.input_path = args.entry_point
    data = data_read_util(args.input_path, args.query_num, args.filter)
    colname = args.input_claim_name_1
    if args.drop_duplicates:
        data = data.drop_duplicates(subset=[colname])
    gens = data[colname]
    likelihood = data[likelihood_colname]
    os.system(f"cp {args.entry_point} {args.base_path}")
    args.output_path = args.base_path + "/" + os.path.basename(args.entry_point)
    return colname, gens, likelihood


def runi(args, colname, generator=None, operation="i"):
    """
    Sample implications for seed documents.
    """
    # Previously saved seed documents are used as input.
    args.input_path = args.output_path

    # Version number of the prompt.
    version = ""
    if operation[-1].isdigit():
        operation, version = operation[:-1], operation[-1]

    # Cache path for the generator.
    args.cache_path = args.cache_path.split(".json")[0] + CACHESUF[operation] + ".json"

    # Output path for the generator.
    args.output_path = (
        args.output_path.split(".json")[0] + "_" + NICK2OPERATION[operation] + ".json"
    )

    # Generate.
    version = "" if version == "" else "_v" + version
    args.operation = NICK2OPERATION[operation] + version
    print("Running implication with mode: ", args.operation)
    args.input_claim_name_1 = colname
    result = query_main(args, generator)
    args.query_num = 9999999
    return result


def runc(args, colname, generator=None, version="c"):
    args.cache_path = args.cache_path.split(".json")[0] + "_cont.json"
    args.input_path = args.output_path
    args.output_path = args.output_path.split(".json")[0] + "_contrasting.json"
    version = version.split("c")[-1]
    version = "" if version == "" else "_v" + version
    args.operation = "similar_contradicting" + version
    args.input_claim_name_1 = colname
    result = query_main(args, generator)

    # Use all previously generated claims as input.
    args.query_num = 9999999
    return result


def runv(args, premise, hypothesis, generator=None, version="vi"):
    args.input_claim_name_1 = premise
    args.input_claim_name_2 = hypothesis
    args.input_path = args.output_path
    if args.operation == "implied":
        args.cache_path = args.cache_path.split(".json")[0] + "_genimpl.json"
    elif args.operation == "contradiction":
        args.cache_path = args.cache_path.split(".json")[0] + "_cont.json"
    version = version.split("c")[-1]
    version = "" if len(version) < 3 else "_v" + version[-1]
    args.operation = args.operation + version

    tempprev = args.temperature
    args.temperature = 0.4
    print("Setting temperature to 0.4 for validity check")
    result = query_main(args, generator)
    args.temperature = tempprev
    return result


def runl(args, colname, generator=None):
    args.input_claim_name_1 = colname
    args.cache_path = args.cache_path.split(".json")[0] + "_likelihood.json"
    args.input_path = args.output_path
    args.operation = "likelihood"
    return query_main(args, generator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--input_path", type=str, help="json lines file")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--input_claim_name_1", type=str, default="claim1")
    parser.add_argument("--input_claim_name_2", type=str, default="claim2")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=12)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--short_gen_len", type=int, default=512)
    parser.add_argument("--long_gen_len", type=int, default=512)
    parser.add_argument("--query_num", type=int, default=999999)
    parser.add_argument("--likelihood_colname", type=str, default="likelihood")
    parser.add_argument("--skip_best_assignment", action="store_true", default=False)
    parser.add_argument("--drop_duplicates", action="store_true", default=False)

    args = parser.parse_args()

    # Save initial set of arguments for next traversal.
    args.entry_point = args.input_path
    args.base_path = args.output_path
    args.init_input_claim_name_1 = args.input_claim_name_1
    args.init_query_num = args.query_num
    args.init_cache_path = args.cache_path

    # Validate graph and identify branches.
    validate_graph(args.graph)
    paths = identify_paths(args.graph)

    # Initialize.
    os.makedirs(args.output_path, exist_ok=True)
    generator = get_llama(args)
    manager = GraphManager()
    gens_stack = []

    # Go through each branch of the tree from root to leaf, DFS.
    # e.g. "r|i,c" --> ["r|i", "r|c"]
    for path in paths:
        # Columns of interest (coi):
        # Tracking the data field name for the last operation
        coi = []
        for hop in path:
            for op in hop.split("-"):
                """
                Glossary:
                b, be: generate bunch of claims, w/ examples
                tg, teg: transductive generation, w/ examples
                r: read from file
                i, fi, demi, demimh: generate implications
                c: generate contradictions
                vi, vc: check validity
                l: compute likelihood
                """
                if op == "b":
                    name, gens = runb(args, generator)
                    gens_stack.append(gens)
                    coi.append(name)
                    manager.add_root_node_batch(gens, kind="root")
                elif op == "be" or op == "bet":
                    name, gens = runbe(args, generator)
                    gens_stack.append(gens)
                    coi.append(name)
                    if op.endswith("t"):
                        fixed = [True] * len(gens)
                        truth_val = [True] * len(gens)
                        likelihood = [1] * len(gens)
                        gens = list(zip(gens, likelihood, truth_val, fixed))
                    manager.add_root_node_batch(gens, kind="root")
                elif op == "tg" or op == "teg":
                    name, gens = runtg(args, generator, extend=op == "teg")
                    gens_stack.append(gens)
                    coi.append(name)
                    manager.add_root_node_batch(gens, kind="root")
                elif op == "r":
                    name, gens, likelihood = runr(args, args.likelihood_colname)
                    gens_stack.append(gens)
                    coi.append(name)
                    fixed = [True] * len(gens)
                    truth_val = [l == 1 for l in likelihood]
                    manager.add_root_node_batch(
                        list(zip(gens, likelihood, truth_val, fixed)), kind="root"
                    )
                elif op.startswith("i") or any(
                    s in op for s in ["fi", "demi", "demimh"]
                ):
                    name, fromgens, togens = runi(
                        args, coi[-1], generator, operation=op
                    )
                    gens_stack.append(togens)
                    coi.append(name)
                    manager.add_leaf_node_batch(fromgens, togens, kind="i")
                elif op.startswith("c"):
                    name, fromgens, togens = runc(args, coi[-1], generator, version=op)
                    gens_stack.append(togens)
                    coi.append(name)
                    manager.add_leaf_node_batch(fromgens, togens, kind="c")
                elif op == "l":
                    probs, gens = runl(args, coi[-1], generator)
                    manager.update_likelihoods(dict(zip(gens, probs)))
                elif "v" in op:
                    args.operation = "implied" if "i" in op else "contradiction"
                    validity, fromgens, togens = runv(
                        args, coi[-2], coi[-1], generator, version=op
                    )
                    fromto = list(zip(fromgens, togens))
                    manager.update_validity(dict(zip(fromto, validity)))
                else:
                    raise ValueError("Invalid operation: " + op)
        # Reset the parameters for the next branch traversal.
        args.output_path = args.base_path
        args.input_path = args.entry_point
        args.input_claim_name_1 = args.init_input_claim_name_1
        args.query_num = args.init_query_num
        args.cache_path = args.init_cache_path

    # Initialize truth values and find the most probable assignment.
    if not args.skip_best_assignment:
        manager.init_truth_values()
        manager.set_best_assignment()

    # Get directory name.
    path = args.base_path + "/all_nodes.csv"
    manager.save_all_nodes(path)
    path = args.base_path + "/all_graphs.txt"
    manager.print_graphs(path)


if __name__ == "__main__":
    print("Running run_alt.py")
    main()
