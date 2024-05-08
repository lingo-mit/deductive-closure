from graph import ClaimNode, ClaimGraph
from typing import List, Dict, Tuple, Union
import pandas as pd
import ipdb


class GraphManager:
    def __init__(self):
        self.seq_to_node: Dict[str, ClaimNode] = {}
        self.node_to_graph: Dict[ClaimNode, ClaimGraph] = {}
        self.graphs: List[ClaimGraph] = []

    def add_root_node(
        self,
        sequence: str,
        prob: float = None,
        truth_val: bool = None,
        fixed: bool = False,
    ):
        if sequence in self.seq_to_node or sequence.isspace():
            return
        node = ClaimNode(sequence, prob, truth_val, fixed)
        self.seq_to_node[sequence] = node
        self.node_to_graph[node] = ClaimGraph(node)
        self.graphs.append(self.node_to_graph[node])

    def add_leaf_node(self, from_sequence, to_sequence, kind, prob: float = None):
        if from_sequence not in self.seq_to_node:
            return
        if (
            from_sequence == to_sequence
            or from_sequence.isspace()
            or to_sequence.isspace()
        ):
            return
        if to_sequence in self.seq_to_node:
            return
        assert from_sequence in self.seq_to_node
        assert to_sequence not in self.seq_to_node
        from_node = self.seq_to_node[from_sequence]
        target_graph = self.node_to_graph[from_node]
        if len(target_graph) >= 10:
            return
        if from_node.truth_value_fixed and from_node.truth_value:
            if kind == "c":
                to_node = ClaimNode(to_sequence, 0, False, True)
            elif kind == "i":
                to_node = ClaimNode(to_sequence, 1, True, True)
            else:
                raise ValueError(f"Invalid kind {kind}")
        else:
            to_node = ClaimNode(to_sequence, prob)
        self.seq_to_node[to_sequence] = to_node
        self.node_to_graph[to_node] = self.node_to_graph[from_node]
        self.node_to_graph[from_node].add_edge(from_node, to_node, kind)

    def update_likelihoods(self, likelihoods: Dict[str, float]):
        for graph in self.graphs:
            graph.update_likelihoods(likelihoods)

    def update_validity(self, validities: Dict[Tuple[str, str], bool]):
        for (fromseq, toseq), valid in validities.items():
            if not valid:
                if toseq not in self.seq_to_node or fromseq not in self.seq_to_node:
                    continue
                if toseq == fromseq:
                    continue
                to_node = self.seq_to_node[toseq]
                graph = self.node_to_graph[to_node]
                graph.remove_node(to_node)
                if graph.is_empty():
                    self.graphs.remove(graph)
                del self.seq_to_node[toseq]
                del self.node_to_graph[to_node]

    def add_root_node_batch(
        self,
        sequences: List[Union[str, Tuple[str, float], Tuple[str, float, bool, bool]]],
        kind: str = "root",
    ):
        for seq in sequences:
            prob = None
            truth_value = None
            fixed = False

            if isinstance(seq, tuple):
                if len(seq) == 4:
                    seq, prob, truth_value, fixed = seq
                elif len(seq) == 2:
                    seq, fixed = seq
                else:
                    raise ValueError()

            self.add_root_node(seq, prob, truth_value, fixed)

    def add_leaf_node_batch(
        self,
        from_sequences: List[str],
        to_sequences: List[Union[str, Tuple[str, float]]],
        kind: str,
    ):
        for from_seq, to_seq in zip(from_sequences, to_sequences):
            prob = None
            if isinstance(to_seq, tuple):
                to_seq, prob = to_seq
            self.add_leaf_node(from_seq, to_seq, kind, prob)

    def init_truth_values(self):
        for graph in self.graphs:
            graph.init_truth_values()

    def set_best_assignment(self):
        for graph in self.graphs:
            comb, prob = graph.compute_joint_probability()
            graph.assign_truth_values(comb)

    def save_all_nodes(self, path: str):
        all_nodes = []
        for graph in self.graphs:
            all_nodes.extend(graph.get_nodes())
        df = pd.DataFrame(all_nodes)
        df.to_csv(path, index=False)

    def print_graphs(self, path: str):
        with open(path, "w") as f:
            for graph in self.graphs:
                f.write(str(graph))
                f.write("\n\n###\n\n")
