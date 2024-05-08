import itertools
from typing import Dict


class ClaimNode:
    def __init__(
        self,
        name,
        likelihood: float = None,
        truth_value: bool = None,
        truth_value_fixed: bool = False
    ):
        self.name = name
        self.likelihood = likelihood
        self.truth_value = truth_value
        self.truth_value_fixed = truth_value_fixed

    def reset_truth_value(self):
        self.set_truth_value(None)

    def set_truth_value(self, truth_value):
        if not self.truth_value_fixed:
            self.truth_value = truth_value
        else:
            print(f"Truth value of {self.name} cannot be set because it is fixed.")

    def set_likelihood(self, likelihood):
        if not self.truth_value_fixed:
            self.likelihood = likelihood
        else:
            print(f"Likelihood of {self.name} cannot be set because its truth value is fixed.")
    
    def __str__(self):
        like = f"{round(self.likelihood, 2)}" if self.likelihood is not None else "None"
        ss = f"{self.name} ({like}) V={self.truth_value}"
        ss = ss + " (fixed)" if self.truth_value_fixed else ss
        return ss

    def __dict__(self):
        return {
            "name": self.name,
            "likelihood": self.likelihood,
            "truth_value": self.truth_value,
            "truth_value_fixed": self.truth_value_fixed
        }


class ClaimGraph:
    def __init__(self, root: ClaimNode):
        self.root = root
        self.edges = {}
        self.nodes = [root]

    def is_empty(self):
        return len(self.nodes) == 0

    def _add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, node1, node2, kind):
        assert kind in ["c", "i"]
        self._add_node(node1)
        self._add_node(node2)
        self.edges[(node1, node2)] = kind  # what if same edge different kinds?

    def _remove_edge(self, node1, node2):
        del self.edges[(node1, node2)]

    def remove_node(self, node):
        self.nodes.remove(node)
        keys = list(self.edges.keys())
        for n1, n2 in keys:
            if n1 == node or n2 == node:
                self._remove_edge(n1, n2)

    def update_likelihoods(self, likelihoods: Dict[str, float]):
        for node in self.nodes:
            if node.name in likelihoods:
                node.set_likelihood(likelihoods[node.name])

    def init_truth_values(self):
        candidate_truth_values = list(itertools.product([True, False], repeat=len(self.nodes)))

        print(f"Number of candidate truth values: {len(candidate_truth_values)}")
        # Eliminate all candidate truth values that are inconsistent with a node's truth value
        for node in self.nodes:
            if node.truth_value == True:
                candidate_truth_values = [combination for combination in candidate_truth_values if combination[self.nodes.index(node)] == True]
            elif node.truth_value == False:
                candidate_truth_values = [combination for combination in candidate_truth_values if combination[self.nodes.index(node)] == False]
        print(f"Number of candidate truth values after marked truth value fix: {len(candidate_truth_values)}")

        incompatible = []
        # Eliminate all candidate truth values that are inconsistent with the graph
        for combination in candidate_truth_values:
            node_val = dict(zip(self.nodes, combination))
            for (n1, n2), kind in self.edges.items():
                if kind == "c" and node_val[n1] == True and node_val[n2] == True:
                    # print(f"Eliminating {combination}")
                    incompatible.append(combination)
                    break
                elif kind == "i" and node_val[n1] == True and node_val[n2] == False:
                    # print(f"Eliminating {combination}")
                    incompatible.append(combination)
                    break
        
        candidate_truth_values = [c for c in candidate_truth_values if c not in incompatible]
        self.cand_truth_values = candidate_truth_values
        print(f"Number of candidate truth values after consistency: {len(candidate_truth_values)}")

    def compute_joint_probability(self):
        comb2prob = {}
        for combination in self.cand_truth_values:
            prob = 1
            for node, val in zip(self.nodes, combination):
                if val is True:
                    prob *= node.likelihood
                else:
                    prob *= (1 - node.likelihood)
            comb2prob[combination] = prob
        self.comb2prob = comb2prob

        # Return the combination with the highest probability
        max_prob = max(comb2prob, key=comb2prob.get)
        self.best_combination = max_prob
        self.best_prob = comb2prob[max_prob]
        return self.best_combination, self.best_prob

    def assign_truth_values(self, combination):
        for node, val in zip(self.nodes, combination):
            node.set_truth_value(val)

    def reset_truth_values(self):
        for node in self.nodes:
            node.reset_truth_value()

    def get_nodes(self):
        # Return a list of dictionaries
        return [node.__dict__() for node in self.nodes]

    def _dfs(self, node, visited):
        visited.add(node)
        result = str(node)
        children = [(n1, n2) for n1, n2 in self.edges.keys() if n1 == node]
        for n1, n2 in children:
            if n2 not in visited:
                result += f"\n{str(n1)} --{self.edges[(n1,n2)]}--> {self._dfs(n2, visited)}"
        return result

    def __str__(self):
        return self._dfs(self.root, set())

    def __len__(self):
        return len(self.nodes)

    def __dict__(self):
        return {
            "root": self.root.__dict__(),
            "edges": self.edges,
            "nodes": [node.__dict__() for node in self.nodes]
        }
