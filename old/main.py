# -*- coding: utf-8 -*-
import numpy as np
import copy
from difflib import SequenceMatcher
import Levenshtein
import time
from tqdm import tqdm
import pandas as pd
import itertools
import random
from typing import List, Tuple, Dict
import networkx as nx
import json
import ast
from dataclasses import dataclass
from pprint import pprint
import argparse
import os
import sys

REPETITIONS = 1
RESULTS_FILE = "nsga_results.csv"
POP_SIZE = 200
GENERATIONS = 200
MAX_NODES = 3
# probabilities used in `mutate`
ADD_NODE_PROB = 0.1
FLIP_EDGE_PROB = 0.3
MODEL_MUTATION_PROB = 0.3

# default query‐types
QUERY_TYPES = [
    'Art', 'Geography', 'History', 'Science and technology', 'Sports']
#     'Music', 'Other', 'Politics',
#     'TV shows', 'Video games',
#     'biology_mmlu', 'business_mmlu', 'chemistry_mmlu', 'computer science_mmlu',
#     'economics_mmlu', 'engineering_mmlu', 'health_mmlu', 'history_mmlu',
#     'law_mmlu', 'math_mmlu', 'other_mmlu', 'philosophy_mmlu',
#     'physics_mmlu', 'psychology_mmlu'
# ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run NSGA-II and/or RandomMOQO")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file")
    parser.add_argument("--algorithm", choices=["both", "nsga", "moqo", "fptas", "all"], default="both",
                        help="Which algorithm to run: both (NSGA-II+RandomMOQO), nsga, moqo, fptas, or all (all three)")
    parser.add_argument("--moqo-time", type=float, default=None,
                        help="Fixed timeout for RandomMOQO in seconds (overrides average NSGA-II time)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Approximation factor for FPTAS (default: 0.1 = 10%% approximation)")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Maximum number of nodes in DAG (default: 3 for normal mode, 5 for experiments)")

    # Experiment mode: vary k and PerfDB levels
    parser.add_argument("--experiment", action="store_true",
                        help="Run full experiment varying k and PerfDB levels")
    parser.add_argument("--k-values", type=str, default="1,2,3,4,5",
                        help="Comma-separated k values to test (default: 1,2,3,4,5)")
    parser.add_argument("--levels", type=str, default="0,1,2,3,4",
                        help="Comma-separated PerfDB levels to test (default: 0,1,2,3,4)")
    parser.add_argument("--level-dir", default="levels",
                        help="Directory containing PerfDB level data files (default: levels)")

    # Diversity experiment: evaluate effect of LLM diversity on QoA and resource usage
    parser.add_argument("--diversity-experiment", action="store_true",
                        help="Run diversity experiment (K=5, vary diversity level d)")
    parser.add_argument("--diversity-levels", type=str, default=None,
                        help="Comma-separated diversity levels to test (default: all from 1 to min(5, |L|))")

    return parser.parse_args()


def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
            # (no overrides previously)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file '{path}': {e}")
        return
    except Exception as e:
        print(f"Error loading config file '{path}': {e}")
        return

    global REPETITIONS, RESULTS_FILE, POP_SIZE, GENERATIONS, MAX_NODES
    global ADD_NODE_PROB, FLIP_EDGE_PROB, MODEL_MUTATION_PROB, QUERY_TYPES
    REPETITIONS = cfg.get("repetitions", REPETITIONS)
    RESULTS_FILE = cfg.get("results_file_name", RESULTS_FILE)
    POP_SIZE = cfg.get("pop_size", POP_SIZE)
    GENERATIONS = cfg.get("generations", GENERATIONS)
    MAX_NODES = cfg.get("max_nodes", MAX_NODES)
    ADD_NODE_PROB = cfg.get("prob_add_node", ADD_NODE_PROB)
    FLIP_EDGE_PROB = cfg.get("prob_flip_edge", FLIP_EDGE_PROB)
    MODEL_MUTATION_PROB = cfg.get("prob_model_mutation", MODEL_MUTATION_PROB)
    QUERY_TYPES = list(cfg.get("query_types", QUERY_TYPES))


# -------------------------
# Global safety/constants
# -------------------------
DAG_DEBUG_PRINT = False
EPS = 1e-9
DEFAULT_QOA = 0.5


def edge_bit_index(i: int, j: int, k: int) -> int:
    """
    Bit ordering: (0,1),(0,2)...(0,k-1),(1,2)...(k-2,k-1)
    Requires i < j.
    """
    if not (0 <= i < j < k):
        raise ValueError(f"edge_bit_index requires 0 <= i < j < k, got i={i}, j={j}, k={k}")
    idx = 0
    for a in range(i):
        idx += (k - 1 - a)
    idx += (j - i - 1)
    return idx


def mask_to_adj(k: int, mask: int):
    adj = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if mask & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1
    return adj


def canonical_mask_and_perm(adj_matrix, k: int):
    """
    Returns (best_mask, best_perm) where:
      - best_perm is a permutation tuple/list of length k.
      - best_perm[new_pos] = old_node_id placed at that new position.
    """
    nodes = list(range(k))
    best_mask = -1
    best_perm = None

    for perm in itertools.permutations(nodes):
        pos = {node: idx for idx, node in enumerate(perm)}
        valid = True
        for i in range(k):
            for j in range(k):
                if adj_matrix[i][j] == 1 and pos[i] > pos[j]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        new_mask = 0
        bit_index = 0
        for a in range(k - 1):
            for b in range(a + 1, k):
                x, y = perm[a], perm[b]
                if adj_matrix[x][y] == 1:
                    new_mask |= (1 << bit_index)
                bit_index += 1

        if new_mask > best_mask:
            best_mask = new_mask
            best_perm = perm

    return best_mask, best_perm

def canonical_representation(adj_matrix, k):
    """Compute highest-value adjacency bitmask among all isomorphic labelings of a DAG."""
    nodes = list(range(k))
    best_mask = -1
    for perm in itertools.permutations(nodes):
        # Check if perm is a valid topological order (no edge goes from later to earlier in this perm)
        valid_topo = True
        pos = {node: idx for idx, node in enumerate(perm)}
        for i in range(k):
            for j in range(k):
                if adj_matrix[i][j] == 1 and pos[i] > pos[j]:
                    valid_topo = False
                    break
            if not valid_topo:
                break
        if not valid_topo:
            continue
        # Build adjacency in new labeling and compute bitmask
        new_mask = 0
        bit_index = 0
        for a in range(k - 1):
            for b in range(a + 1, k):
                # Map original nodes corresponding to new labels a, b
                # Find original nodes x,y such that perm.index(x)=a, perm.index(y)=b
                x = perm[a];
                y = perm[b]
                if adj_matrix[x][y] == 1:
                    new_mask |= (1 << bit_index)
                bit_index += 1
        if new_mask > best_mask:
            best_mask = new_mask
    return best_mask


def is_valid_dag(k, mask):
    """
    Check if a bitmask represents a valid DAG:
    - Single sink (only highest node has out-degree 0)
    - Fully connected (all nodes reach sink)

    Returns: True if valid, False otherwise
    """
    if k > 1 and mask == 0:
        return False  # Empty graph

    # Reconstruct adjacency matrix from bitmask
    adj = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if mask & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1

    # Check one-sink condition
    outdeg = [0] * k
    for i in range(k):
        for j in range(k):
            if adj[i][j] == 1:
                outdeg[i] += 1

    # Only highest node should have out-degree 0
    if any(outdeg[node] == 0 for node in range(k - 1)):
        return False

    # Check full connectivity (every node can reach the sink)
    reachable = [False] * k
    reachable[-1] = True  # mark sink
    # BFS backward from sink
    queue = [k - 1]
    while queue:
        cur = queue.pop(0)
        for prev in range(cur):
            if adj[prev][cur] == 1 and not reachable[prev]:
                reachable[prev] = True
                queue.append(prev)

    return all(reachable)


def generate_nonisomorphic_dags(max_nodes):
    """Generate all fully-connected DAGs (up to max_nodes) with one sink, returning unique canonical structures."""
    unique_dags = []
    seen_structs = set()
    for k in range(1, max_nodes + 1):
        num_edges = k * (k - 1) // 2  # number of possible edges in upper triangle
        for mask in range(1 << num_edges):
            # Check if valid DAG
            if not is_valid_dag(k, mask):
                continue

            # Reconstruct adjacency for canonicalization
            adj = [[0] * k for _ in range(k)]
            bit_index = 0
            for i in range(k - 1):
                for j in range(i + 1, k):
                    if mask & (1 << bit_index):
                        adj[i][j] = 1
                    bit_index += 1

            # Compute canonical structure id for this DAG
            canon_mask = canonical_representation(adj, k)
            if (k, canon_mask) not in seen_structs:
                # print(f"k={k}, mask={canon_mask}, adj = {adj}")
                seen_structs.add((k, canon_mask))
                unique_dags.append((k, canon_mask))
    return unique_dags


def random_valid_structure(k, max_attempts=100, skip_canonical=False):
    """
    Generate a random valid DAG structure without enumeration.

    Args:
        k: Number of nodes
        max_attempts: Maximum attempts before fallback
        skip_canonical: If True, skip canonicalization (faster but may have duplicates)

    Returns:
        (k, struct_id) tuple (canonical if skip_canonical=False)
    """
    num_edges = k * (k - 1) // 2

    for attempt in range(max_attempts):
        # Generate random bitmask (ensure at least one edge for k>1)
        if k == 1:
            mask = 0  # No edges for single node
        else:
            # Ensure at least one edge by setting random bits
            mask = 0
            # Set at least k-1 random edges to ensure connectivity chance
            num_bits = random.randint(k - 1, num_edges)
            for _ in range(num_bits):
                bit_pos = random.randint(0, num_edges - 1)
                mask |= (1 << bit_pos)

        # Check if valid
        if is_valid_dag(k, mask):
            if skip_canonical:
                # Skip expensive canonicalization for large k
                return k, mask
            else:
                # Reconstruct adjacency for canonicalization
                adj = [[0] * k for _ in range(k)]
                bit_index = 0
                for i in range(k - 1):
                    for j in range(i + 1, k):
                        if mask & (1 << bit_index):
                            adj[i][j] = 1
                        bit_index += 1

                # Return canonical representation
                canonical = canonical_representation(adj, k)
                return k, canonical

    # Fallback: return simple sequential structure (0→1→2→...→k-1)
    # This is always valid
    mask = 0
    bit_index = 0
    for i in range(k - 1):
        # Connect node i to node i+1
        for j in range(i + 1, k):
            if j == i + 1:
                mask |= (1 << bit_index)
            bit_index += 1

    if skip_canonical:
        return k, mask

    adj = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if mask & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1

    canonical = canonical_representation(adj, k)
    return k, canonical


def assign_models_to_dag(k, struct_id):
    # Reconstruct adjacency to get in-degrees
    def adjacency_from_mask(k, mask):
        adj_list = {i: [] for i in range(k)}
        bit_index = 0
        for i in range(k - 1):
            for j in range(i + 1, k):
                if mask & (1 << bit_index):
                    adj_list[i].append(j)
                bit_index += 1
        return adj_list

    def indegree_list(k, adj_list):
        indeg = [0] * k
        for u in adj_list:
            for v in adj_list[u]:
                indeg[v] += 1
        return indeg

    adj_list = adjacency_from_mask(k, struct_id)
    indeg = indegree_list(k, adj_list)
    assignment = []
    for node in range(k):
        if indeg[node] > 1:
            assignment.append(5)  # blending model for merge nodes
        else:
            assignment.append(random.randint(0, 4))  # random base LLM for others
    return assignment


def adjacency_matrix_from_mask(mask, k):
    """Generates an adjacency matrix from a given mask and number of nodes.
    Args:
        mask: An integer representing the connections in the graph.
        k: The number of nodes in the graph.
    Returns:
        A list of lists representing the adjacency matrix.
    """
    adj_matrix = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if (mask >> bit_index) & 1:
                adj_matrix[i][j] = 1
            bit_index += 1
    return adj_matrix


def evaluate_individual_V2(struct_id, assignment, query_type, query_tokens, blending_prompt_tokens, ctx_tokens,
                           df_history):
    # print(f'Evaluating individual structure_id: {struct_id} assignment: {assignment}')
    k = len(assignment)
    # Reconstruct adjacency list from struct_id
    adj_list = {i: [] for i in range(k)}
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if struct_id & (1 << bit_index):
                adj_list[i].append(j)
            bit_index += 1

    llm_assignment = []
    for model in assignment:
        llm_assignment.append(str(model))

    # print('adj list')
    # pprint(adj_list)

    adjacency_matrix = adjacency_matrix_from_mask(struct_id, k)
    matrix = np.array(adjacency_matrix)
    size = matrix.shape[0]
    # print('matrix', adjacency_matrix)

    # Create directed graph
    G = nx.DiGraph()
    for idx, node_dict in enumerate(llm_assignment):
        G.add_node(idx)
        G.nodes[idx]['info'] = node_dict

    # Add edges based on the upper triangular adjacency matrix
    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i][j] == 1:
                # print(f'addding edge {i} to {j}')
                G.add_edge(i, j)

    # 3-tier evaluation: exact PerfDB → blend operation → fuzzy estimator
    final_metrics = None

    # Tier 1: Try exact PerfDB lookup
    final_metrics = get_subdag_metrics_v7(G, llm_assignment, query_type, df_history)

    # Tier 2: Try blend operation lookup
    if final_metrics is None:
        final_metrics = special_get_subdag_metrics_for_one_blend_operations(G, llm_assignment, query_type, df_history)

    # Tier 3: Fallback to fuzzy matching estimator
    if final_metrics is None:
        final_metrics = estimate_schedule_v3(
            G,
            llm_assignment,
            query_type,
            query_tokens,
            blending_prompt_tokens,
            ctx_tokens,
            df_history,
            levenshtein_threshold=0.75,
            turn_off_exact_fuzzy_matching=True  # Match FPTAS: no fuzzy matching
        )

    # global_debug_graph.append((struct_id, assignment))
    return (final_metrics.final_cost, final_metrics.final_latency, final_metrics.final_energy,
            final_metrics.quality_of_answer)


## --- newly added
def get_adj_from_graph(G: nx.DiGraph):
    node_order = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=node_order, dtype=int)
    return adj_matrix


## --- newly added
def canonical_label_for_isomorphism(G: nx.DiGraph):
    adj_matrix = get_adj_from_graph(G)
    return canonical_representation(adj_matrix, len(G.nodes()))


def construct_subdag_behind_node(
        whole_graph: nx.DiGraph,
        sink_node: int,
        llm_assignment: List[str]
):
    """
    The idea here is to construct an entire graph behind a single node
    - We know that we are dealing with DAGs, so there are no cycles
    - given some @sink_node in @whole_graph, we can construct a subgraph
      where every node and the subgraph's connections lead up to the @sink_node

    Approach:
    - Iterative DFS where we start with @sink_node, and add @sink_node's predecessors to
      subgraph, then the process is repeated by adding the predecessors's predecessors to the graph
    - Effectively, build the subgraph
    """
    # stack to emulate recursion
    stack = [sink_node]
    subgraph = nx.DiGraph()
    seen = set()
    while stack:
        node = stack.pop(-1)
        if node in seen:
            continue
        subgraph.add_node(node)
        seen.add(node)

        predecessors = list(whole_graph.predecessors(node))
        for pred in predecessors:
            subgraph.add_edge(pred, node)
            stack.append(pred)

    # Get the proper llm assignment by using the node values as an index into the llm_assignment list
    subgraph_llm_assignment = [llm_assignment[node] for node in sorted(subgraph.nodes())]
    return (subgraph, subgraph_llm_assignment, sink_node)


def get_sub_llm_assignment(subDAG, llm_assignment):
    return [llm_assignment[node] for node in sorted(subDAG.nodes())]


def get_sub_llm_assignment_given_nodes(nodes, llm_assignment):
    return [llm_assignment[node] for node in sorted(nodes)]


def is_blend_node(G: nx.DiGraph, node: int) -> bool:
    return G.in_degree(node) >= 2


def is_sequential_node(G: nx.DiGraph, node: int) -> bool:
    return G.in_degree(node) == 1


def is_start_node(G: nx.DiGraph, node: int) -> bool:
    return G.in_degree(node) == 0


def is_sink(G1: nx.DiGraph, G2: nx.DiGraph, node: int) -> bool:
    return G1.out_degree(node) == 0


@dataclass
class LLMMetrics:
    input_cost: float = 0.0
    input_latency: float = 0.0
    input_energy: float = 0.0
    output_cost: float = 0.0
    output_latency: float = 0.0
    output_energy: float = 0.0
    quality_of_answer: float = 0.0
    average_output_tokens: float = 0.0
    final_cost: float = 0.0
    final_latency: float = 0.0
    final_energy: float = 0.0


@dataclass
class BFSNode:
    LLM_name: str
    node: int
    metrics: LLMMetrics  # e.g., cost, latency, energy, qoa
    accumulated_metrics: LLMMetrics  # Accumulated along the path
    first_node_in_seq: bool

    def __iter__(self):
        return iter((self.LLM_name, self.node, self.metrics, self.accumulated_metrics, self.first_node_in_seq))


@dataclass
class ProcessingTableEntry:
    LLM: str
    node: int
    metrics: LLMMetrics
    accumulated_metrics: LLMMetrics


@dataclass
class BlendingNodeDBReference:
    inputs: Dict[str, float]
    output: float = 0.0


def special_get_subdag_metrics_for_one_blend_operations(
  subdag: nx.DiGraph,
  sub_assignment: List[str],
  query_type: str,
  df_history: pd.DataFrame
):
  """
  Improved version with proper structure_id comparison.
  Allows multiple rows; returns mean metrics.
  """
  adj_matrix = get_adj_from_graph(subdag)
  structure_id = canonical_representation(adj_matrix, len(sub_assignment))

  # Build all valid permutations for the non-sink inputs when there's a single blend op
  all_valid_forms = []
  if len(sub_assignment) >= 2:
    llms = sub_assignment[0:-1]
    perms = set(itertools.permutations(llms))
    for perm in perms:
      all_valid_forms.append(",".join(list(perm) + [sub_assignment[-1]]))
  else:
    all_valid_forms.append(",".join(sub_assignment))

  possible = df_history[
    (df_history["structure_id"] == structure_id) &
    (df_history["llm_assignments"].isin(all_valid_forms)) &
    (df_history["query_type"] == query_type)
  ]

  if len(possible) == 0:
    return None

  return LLMMetrics(
    input_cost=possible["input_cost"].mean(),
    input_latency=possible["input_latency"].mean(),
    input_energy=possible["input_energy"].mean(),
    output_cost=possible["output_cost"].mean(),
    output_latency=possible["output_latency"].mean(),
    output_energy=possible["output_energy"].mean(),
    quality_of_answer=possible["qoa"].mean(),
    average_output_tokens=possible["average_output_tokens"].mean(),
    final_cost=possible["cost"].mean(),
    final_latency=possible["latency"].mean(),
    final_energy=possible["energy"].mean(),
  )



def get_subdag_metrics_v7(
        subdag: nx.DiGraph,
        sub_assignment: List[str],
        query_type: str,
        df_history: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    """
    Improved version with proper structure_id comparison
    """
    adj_matrix = get_adj_from_graph(subdag)  # construct
    structure_id = canonical_representation(adj_matrix, len(sub_assignment))
    # print('sub assignment', sub_assignment)
    assignment_str = ",".join(sub_assignment)
    # print("structure id for match", structure_id)
    # print("assignment str for match", assignment_str)

    possible = df_history[
        (df_history["structure_id"] == structure_id) &
        (df_history["llm_assignments"] == assignment_str) &
        (df_history["query_type"] == query_type)
        ]

    if len(possible) == 0:
        return None
    else:
        ### final_cost, final_latency, final_energy will be used for matched subschedules
        ### fields like input_cost and output_cost are used for estimation purposes
        return LLMMetrics(
            input_cost=possible["input_cost"].mean(),
            input_latency=possible["input_latency"].mean(),
            input_energy=possible["input_energy"].mean(),
            output_cost=possible["output_cost"].mean(),
            output_latency=possible["output_latency"].mean(),
            output_energy=possible["output_energy"].mean(),
            quality_of_answer=possible["qoa"].mean(),
            average_output_tokens=possible["average_output_tokens"].mean(),
            final_cost=possible["cost"].mean(),
            final_latency=possible["latency"].mean(),
            final_energy=possible["energy"].mean(),
        )


def get_single_model_metrics(llm_assignment: List[str], query_type: str, df_history: pd.DataFrame):
    single_models = {}
    for llm in llm_assignment:
        single_model_DAG = nx.DiGraph()
        single_model_DAG.add_node(llm)
        single_model_metrics = get_subdag_metrics_v7(
            single_model_DAG,
            [llm],
            query_type,
            df_history
        )
        if single_model_metrics is not None:
            single_models[llm] = single_model_metrics
    return single_models


def get_pairwise_metrics(llm_assignment: List[str], query_type: str, df_history: pd.DataFrame):
  pairwise_models = {}
  pairwise_DAG = nx.DiGraph()
  pairwise_DAG.add_nodes_from([0, 1])
  pairwise_DAG.add_edge(0, 1)

  for i in range(len(llm_assignment)):
    for j in range(len(llm_assignment)):
      node_i = str(llm_assignment[i]).strip()
      node_j = str(llm_assignment[j]).strip()

      # skip only (5,5)
      if node_i == "5" and node_j == "5":
        continue

      metrics = get_subdag_metrics_v7(
        pairwise_DAG,
        [node_i, node_j],
        query_type,
        df_history
      )
      if metrics is not None:
        pairwise_models[(int(node_i), int(node_j))] = metrics

  return pairwise_models



def convert_metrics_to_dict(metrics):
    return {
        "input_cost": metrics[0],
        "input_latency": metrics[1],
        "input_energy": metrics[2],
        "output_cost": metrics[3],
        "output_latency": metrics[4],
        "output_energy": metrics[5],
        "quality_of_answer": metrics[6],
        "average_output_tokens": metrics[7],
        "final_cost": metrics[8],
        "final_latency": metrics[9],
        "final_energy": metrics[10],
    }


# not used currently in estimation code flow
def fallback_estimation_scheme_v2(
        G: nx.DiGraph,
        assignment: List[str],
        query_type: str,
        df_history: pd.DataFrame,
        get_output_as_dict=False,
        print_debug_log=False
):
    debug_log = []
    topo_order = list(nx.topological_sort(G))
    blend_nodes = list(filter(lambda x: G.in_degree(x) > 1, topo_order[::-1]))
    last_blend_node = blend_nodes[0]
    # get all the inputs for the last blend node
    # if one of the inputs itself is another blend node, get inputs for that as well
    inputs = set()
    stack = [last_blend_node]
    while stack:
        node = stack.pop(-1)
        if node in inputs: continue
        if is_blend_node(G, node):
            stack.extend(list(G.predecessors(node)))
        else:
            inputs.add(node)

    debug_log.append(f'All predecessors input of nested blending node search: {inputs}')
    subDAG = nx.DiGraph()
    subDAG.add_nodes_from(list(inputs) + [last_blend_node])
    for node in inputs:
        if node == last_blend_node: continue
        subDAG.add_edge(node, last_blend_node)

    sub_llm_assignment = get_sub_llm_assignment(subDAG, assignment)
    debug_log.append(f'nodes of the new constructed subDAG for parallel-fallback-match: {subDAG.nodes()}')
    debug_log.append(f'structure_id: {canonical_label_for_isomorphism(subDAG)}')
    debug_log.append(f'llm assignment: {sub_llm_assignment}')

    metrics = get_subdag_metrics_v7(subDAG, sub_llm_assignment, query_type, df_history)
    debug_log.append(f'metrics: {metrics}')
    if print_debug_log:
        print('\n'.join(debug_log))
    return metrics, last_blend_node


def sequential_delta(node_A, node_B, single_model_metrics, pairwise_metrics):
  a = int(node_A)
  b = int(node_B)

  pair = pairwise_metrics.get((a, b))
  if pair is None:
    return 0.0

  final = pair.quality_of_answer
  initial = single_model_metrics[b].quality_of_answer
  initial = initial if initial and initial > 0 else DEFAULT_QOA

  return (final - initial) / max(initial, EPS)



def fuzzy_ranked_matches_from_database(
        llm_assignment: List[str],
        target_structure_id: int,
        target_last_node: str,
        query_type: str,
        df_history: pd.DataFrame,
        threshold: float = 0.7,
        w_seq: float = 0.33,
        w_lev: float = 0.33,
        w_jac: float = 0.33,
        bonus_for_last_node: float = 0.01
) -> List[Tuple[List[str], Tuple[float, float, float, float], float]]:
    target_seq_str = ",".join(llm_assignment)
    target_struct_str = str(target_structure_id)
    # print('target_seq_str',target_seq_str)
    # print('target_struct_str', target_struct_str) # in full form
    best_score = 0.0
    best_row = None
    best_candidate = None

    """
    Note to self:
    cand: list
    cand_str: string representation of cand, but remove brackets and double-quotes
    struct_str: string representation of canonical label
    target_seq_str: string representation of llm_assignment, must be same format at cand_str
    target_struct_str: string representation of canonical label, must be same format at struct_str
    """

    # OPTIMIZATION: Pre-filter DataFrame to reduce iterations from 47K+ to a small subset
    filtered_df = df_history[
        (df_history["query_type"] == query_type) &
        (df_history["structure_id"].astype(str) == target_struct_str)
    ]

    # Early exit if no matching structures found
    if filtered_df.empty:
        return (None, None, best_score)

    for _, row in filtered_df.iterrows():
        cand_str = row["llm_assignments"]
        cand = [item.strip() for item in cand_str.split(',')]

        if not (cand and cand[-1] == target_last_node):
            continue

        # compute hybrid components
        # print(f'llm_assignment: {llm_assignment}, {type(llm_assignment)}, cand {cand}, {type(cand)}, cand_str: {cand_str}, struct_str: {struct_str} target_struct_str:{target_struct_str}')
        # make sure cand is a list, and cand_str is string representation without bracket and quotes to wrap the LLMs
        seq_sim = SequenceMatcher(None, llm_assignment, cand).ratio()
        lev_sim = Levenshtein.ratio(target_seq_str, cand_str)
        jac_sim = len(set(llm_assignment) & set(cand)) / len(set(llm_assignment) | set(cand)) if llm_assignment else 1.0
        # struct_sim = 1.0 if struct_str == target_struct_str else 0.0
        # last_node_eq = 1.0 if cand and cand[-1] == target_last_node else 0.0

        hybrid = w_seq * seq_sim + w_lev * lev_sim + w_jac * jac_sim
        # if cand and cand[-1] == target_last_node:
        #   hybrid += bonus_for_last_node

        if hybrid > best_score:
            best_score = hybrid
            best_row = row
            best_candidate = cand_str

            # OPTIMIZATION: Early termination if we find a near-perfect match
            if best_score >= 0.95:
                break

    # print('made it here')
    # print('best_score', best_score)
    # print('best_row', best_row)
    # print('best_candidate', best_candidate)

    if best_row is not None and best_score >= threshold:
        return (
            best_candidate,
            (best_row["cost"], best_row["latency"],
             best_row["energy"], best_row["qoa"]),
            best_score
        )
    return (None, None, best_score)


def fuzzy_match_sequential_chain(graph: nx.DiGraph, whole_llm_assignment: List[str], query_type: str,
                                 single_model_metrics, df_history: pd.DataFrame, levin_threshold):
    # Assume G is fully sequential
    G = graph.copy()
    llm_assignment = whole_llm_assignment.copy()
    # print('passed in llm_assignment', whole_llm_assignment)
    topo_sort = list(nx.topological_sort(G))
    metrics = None
    # print('in fuzzy')
    # print('plot of whole subgraph under match consideration', llm_assignment)
    # plot_graph(G)
    MIN_LENGTH_OF_SEQUENCE_TO_MATCH = 2
    for i, start_node in enumerate(topo_sort):
        # print('matching', llm_assignment)
        # plot_graph(G)
        # print('i',i, 'sn:', start_node)
        # print('llm assignmnet', llm_assignment)
        if len(topo_sort) - i < MIN_LENGTH_OF_SEQUENCE_TO_MATCH:
            break
        # print(start_node, llm_assignment)
        target_last_node = llm_assignment[-1]
        # print('target_last_node', target_last_node)
        target_structure_id = canonical_label_for_isomorphism(G)
        _, match, score = fuzzy_ranked_matches_from_database(
            llm_assignment,
            target_structure_id,
            target_last_node,
            query_type,
            df_history,
        )
        # print('levin_threshold', levin_threshold)
        if match and score > levin_threshold:
            # print('fuzzy match found')
            metrics = LLMMetrics(average_output_tokens=single_model_metrics[target_last_node].average_output_tokens,
                                 final_cost=match[0], final_latency=match[1], final_energy=match[2],
                                 quality_of_answer=match[3])
            break
        G.remove_node(start_node)
        llm_assignment.pop(0)

    return metrics, G, topo_sort[-1]


# NEW
# calculate the input and output costs of node and its successor, if the connection is sequential
# if node is first in line of sequential chain, then this function will calculate the input and output costs of Node, and then do the same for the successor
# if node is not the first, then all we do is calculate the successors incurred costs, and the calling function will add these costs onto existing metrics
def calculate_cost_sequential_v2(
        query_tokens: int,
        node_A_avg_tokens: int,
        ctx_prompt_tokens: int,
        node_B_avg_tokens: int,
        cost_factors_A: LLMMetrics,
        cost_factors_B: LLMMetrics,
        first_node_in_seq: bool,
):
    if first_node_in_seq:
        # print('in here')
        input_tokens_A = query_tokens
        output_tokens_A = node_A_avg_tokens
        input_tokens_B = ctx_prompt_tokens + node_A_avg_tokens + query_tokens
        output_tokens_B = node_B_avg_tokens
        # print('input tokens_A', input_tokens_A, 'output_tokens_A', output_tokens_A)
        # print('input tokens_B', input_tokens_B, 'output_tokens_B', output_tokens_B)
        # print('input cost factor A', cost_factors_A.input_cost, 'output', cost_factors_A.output_cost, type(cost_factors_A.output_cost) )
        # print('input cost factor B', cost_factors_B.input_cost, 'output', cost_factors_B.output_cost )
        cost = (input_tokens_A * cost_factors_A.input_cost + output_tokens_A * cost_factors_A.output_cost) + (
                    input_tokens_B * cost_factors_B.input_cost + output_tokens_B * cost_factors_B.output_cost)
        latency = (input_tokens_A * cost_factors_A.input_latency + output_tokens_A * cost_factors_A.output_latency) + (
                    input_tokens_B * cost_factors_B.input_latency + output_tokens_B * cost_factors_B.output_latency)
        energy = (input_tokens_A * cost_factors_A.input_energy + output_tokens_A * cost_factors_A.output_energy) + (
                    input_tokens_B * cost_factors_B.input_energy + output_tokens_B * cost_factors_B.output_energy)
        # print('final cost', cost)
        return cost, latency, energy
    else:
        input_tokens_B = ctx_prompt_tokens + node_A_avg_tokens + query_tokens
        output_tokens_B = node_B_avg_tokens
        cost = cost_factors_B.input_cost * input_tokens_B + cost_factors_B.output_cost * output_tokens_B
        energy = cost_factors_B.input_energy * input_tokens_B + cost_factors_B.output_energy * output_tokens_B
        latency = cost_factors_B.input_latency * input_tokens_B + cost_factors_B.output_latency * output_tokens_B
        return cost, latency, energy


def calculate_cost_parallel(
    llm_assignments,
    blend_node,
    current_metrics,
    blending_node_metrics,
    processing_table_entries,
    blending_reference_table,
    single_model_metrics,
    query_tokens,
    blending_prompt_tokens,
    turn_off_blend_table_check=False,
    G=None
):
    current_metrics = copy.deepcopy(current_metrics)

    use_blending_reference_table_entry = True
    if blend_node not in blending_reference_table or turn_off_blend_table_check:
      use_blending_reference_table_entry = False

    blending_reference_table_entry = None
    if use_blending_reference_table_entry:
      blending_reference_table_entry = blending_reference_table[blend_node]
      # ensure no input has <=0 QoA
      for node_idx, inp_metrics in blending_reference_table_entry.inputs.items():
          if inp_metrics.quality_of_answer is None or inp_metrics.quality_of_answer <= 0:
              inp_metrics.quality_of_answer = DEFAULT_QOA

    ncost, nlatency, nenergy, delta = 0.0, 0.0, 0.0, 0.0
    output_tokens = single_model_metrics[ llm_assignments[blend_node] ].average_output_tokens
    input_tokens  = query_tokens + blending_prompt_tokens

    # Determine dominated entries: if entry i's source node is an ancestor of
    # entry j's source node, then i's cost is already accumulated in j's path.
    dominated = set()
    if G is not None:
        entry_nodes = [entry.node for entry in processing_table_entries]
        for i, ni in enumerate(entry_nodes):
            for j, nj in enumerate(entry_nodes):
                if i != j and nx.has_path(G, ni, nj):  # ni is ancestor of nj
                    dominated.add(i)
                    break

    for i, entry in enumerate(processing_table_entries):
        input_tokens += entry.metrics.average_output_tokens

        if i not in dominated:
            ncost    += entry.accumulated_metrics.final_cost
            nenergy  += entry.accumulated_metrics.final_energy
        nlatency  = max(nlatency, entry.accumulated_metrics.final_latency)

        final = entry.accumulated_metrics.quality_of_answer

        if use_blending_reference_table_entry:
          ref = blending_reference_table_entry.inputs.get(entry.node)
          initial = (ref.quality_of_answer if (ref and ref.quality_of_answer and ref.quality_of_answer > 0) else DEFAULT_QOA)
          delta += (final - initial) / max(initial, EPS)

    token_cost    = blending_node_metrics.input_cost    * input_tokens + blending_node_metrics.output_cost    * output_tokens
    token_latency = blending_node_metrics.input_latency * input_tokens + blending_node_metrics.output_latency * output_tokens
    token_energy  = blending_node_metrics.input_energy  * input_tokens + blending_node_metrics.output_energy  * output_tokens

    ncost    += token_cost
    nlatency += token_latency
    nenergy  += token_energy

    if use_blending_reference_table_entry:
      current_metrics.quality_of_answer = blending_reference_table_entry.output * (1 + delta / max(len(processing_table_entries), 1))
      current_metrics.quality_of_answer = min(current_metrics.quality_of_answer, 1.0)  # Clamp to [0, 1]
    else:
      current_metrics.quality_of_answer = DEFAULT_QOA

    current_metrics.final_cost    = ncost
    current_metrics.final_latency = nlatency
    current_metrics.final_energy  = nenergy
    return current_metrics



# NEW
def new_traversal_v3(
    G: nx.DiGraph,
    pairwise_metrics,
    single_model_metrics,
    matched_nodes_information,
    blending_node_db_references,
    query_tokens:int,
    ctx_prompt_tokens:int,
    blending_prompt_tokens:int,
    whole_graph_llm_assignment: List[str]
):
  import collections
  queue = collections.deque()
  processing_table = collections.defaultdict(list)
  processed_sinks = set()

  single_model_metrics = {int(k): v for k, v in single_model_metrics.items()}
  blending_node_db_references = {int(k): v for k, v in blending_node_db_references.items()}
  pairwise_metrics = {(int(a), int(b)): v for (a, b), v in pairwise_metrics.items()}

  whole_graph_llm_assignment = [int(x) for x in whole_graph_llm_assignment]
  llm_assignments = {node: whole_graph_llm_assignment[node] for node in G.nodes()}
  final_metrics = None

  start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

  # EDGE CASE: single node graph
  if len(start_nodes) == 1 and G.out_degree(start_nodes[0]) == 0:
    only = start_nodes[0]
    metrics = single_model_metrics[llm_assignments[only]]
    return LLMMetrics(
        final_cost=metrics.final_cost,
        final_latency=metrics.final_latency,
        final_energy=metrics.final_energy,
        quality_of_answer=metrics.quality_of_answer
    )

  # init queue
  for start_node in start_nodes:
    key = llm_assignments[start_node]
    metrics = copy.deepcopy(single_model_metrics[key])
    metrics.final_cost = 0
    metrics.final_latency = 0
    metrics.final_energy = 0
    queue.append(BFSNode(
        LLM_name=key,
        node=start_node,
        metrics=metrics,
        accumulated_metrics=metrics,
        first_node_in_seq=True
    ))

  while queue:
    bfs_node = queue.popleft()
    LLM_name, node, metrics, current_metrics, first_node_in_seq = bfs_node

    if is_blend_node(G, node) and node not in matched_nodes_information:
      if len(processing_table[node]) != G.in_degree(node):
        queue.append(bfs_node)
        continue
      else:
        current_metrics = calculate_cost_parallel(
            llm_assignments,
            node,
            current_metrics,
            metrics,
            processing_table[node],
            blending_node_db_references,
            single_model_metrics,
            query_tokens,
            blending_prompt_tokens,
            G=G
        )

    successors = list(G.successors(node))
    if len(successors) == 0 and len(queue) == 0:
      final_metrics = current_metrics
      break

    for successor in successors:
      node_key, successor_key = llm_assignments[node], llm_assignments[successor]

      if successor in matched_nodes_information and successor in processed_sinks:
        continue

      if successor in matched_nodes_information:
        processed_sinks.add(successor)
        m = matched_nodes_information[successor][0]
        if isinstance(m, tuple):
          m = convert_metrics_to_dict(m)
        queue.append(BFSNode(
          LLM_name=successor_key,
          node=successor,
          metrics=single_model_metrics[successor_key],
          accumulated_metrics=m,
          first_node_in_seq=False
        ))
        continue

      if is_blend_node(G, successor):
        ncurrent_metrics = copy.deepcopy(current_metrics)

        if is_start_node(G, node):
          cost = query_tokens * metrics.input_cost + metrics.average_output_tokens * metrics.output_cost
          latency = query_tokens * metrics.input_latency + metrics.average_output_tokens * metrics.output_latency
          energy = query_tokens * metrics.input_energy + metrics.average_output_tokens * metrics.output_energy
          ncurrent_metrics.final_cost = cost
          ncurrent_metrics.final_latency = latency
          ncurrent_metrics.final_energy = energy

        processing_table[successor].append(
          ProcessingTableEntry(node=node, LLM=node_key, metrics=metrics, accumulated_metrics=ncurrent_metrics)
        )

        if successor not in processed_sinks:
          processed_sinks.add(successor)
          queue.append(BFSNode(
            LLM_name=successor_key,
            node=successor,
            metrics=single_model_metrics[successor_key],
            accumulated_metrics=single_model_metrics[successor_key],
            first_node_in_seq=False
          ))

      else:
        # sequential successor
        if first_node_in_seq:
          # skip only (5,5)
          if not (node_key == 5 and successor_key == 5):
            ncurrent_metrics = copy.deepcopy(current_metrics)

            pair = pairwise_metrics.get((node_key, successor_key))
            if pair is not None:
              ncurrent_metrics.quality_of_answer = pair.quality_of_answer
            else:
              # fallback: keep current QoA or use successor QoA
              ncurrent_metrics.quality_of_answer = current_metrics.quality_of_answer or single_model_metrics[successor_key].quality_of_answer

            cost, latency, energy = calculate_cost_sequential_v2(
                query_tokens,
                metrics.average_output_tokens,
                ctx_prompt_tokens,
                single_model_metrics[successor_key].average_output_tokens,
                metrics,
                single_model_metrics[successor_key],
                True
            )
            ncurrent_metrics.final_cost = cost
            ncurrent_metrics.final_latency = latency
            ncurrent_metrics.final_energy = energy

            queue.append(BFSNode(
              LLM_name=successor_key,
              node=successor,
              metrics=single_model_metrics[successor_key],
              accumulated_metrics=ncurrent_metrics,
              first_node_in_seq=False
            ))
        else:
          successor_metrics = single_model_metrics[successor_key]
          ncurrent_metrics = copy.deepcopy(current_metrics)

          if is_blend_node(G, node):
            N = 0
            delta = 0.0
            for nn in G.predecessors(node):
              if G.in_degree(nn) >= 2:
                continue
              N += 1
              delta += sequential_delta(llm_assignments[nn], llm_assignments[successor], single_model_metrics, pairwise_metrics)
            qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + (delta / max(N, 1)))
          else:
            qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + sequential_delta(llm_assignments[node], llm_assignments[successor], single_model_metrics, pairwise_metrics))

          # Clamp QoA to [0, 1]
          qoa_estimation = min(qoa_estimation, 1.0)

          cost, latency, energy = calculate_cost_sequential_v2(
              query_tokens,
              single_model_metrics[node_key].average_output_tokens,
              ctx_prompt_tokens,
              successor_metrics.average_output_tokens,
              None,
              successor_metrics,
              False,
          )
          ncurrent_metrics.final_cost += cost
          ncurrent_metrics.final_latency += latency
          ncurrent_metrics.final_energy += energy
          ncurrent_metrics.quality_of_answer = qoa_estimation

          queue.append(BFSNode(
            LLM_name=successor_key,
            node=successor,
            metrics=successor_metrics,
            accumulated_metrics=ncurrent_metrics,
            first_node_in_seq=False
          ))

  return final_metrics



def new_traversal_just_for_cost_latency_energy_v3(
        G: nx.DiGraph,
        pairwise_metrics,  # v6
        single_model_metrics,  # v6
        matched_nodes_information,  # v6
        blending_node_db_references,
        query_tokens: int,
        ctx_prompt_tokens: int,
        blending_prompt_tokens: int,
        whole_graph_llm_assignment: List[str]
):
    # print('in new_traversal_just_for_cost_latency_energy_v3')
    # plot_graph(G)
    import collections
    queue = collections.deque()
    processing_table = collections.defaultdict(list)
    processed_sinks = set()
    single_model_metrics = {
        int(k): v for k, v in single_model_metrics.items()
    }
    blending_node_db_references = {
        int(k): v for k, v in blending_node_db_references.items()
    }
    pairwise_metrics = {
        (int(a), int(b)): v
        for (a, b), v in pairwise_metrics.items()
    }

    # print('single_model_metrics')
    # pprint(single_model_metrics)

    # ─── EDIT B: coerce assignment list entries to int ───
    whole_graph_llm_assignment = [int(x) for x in whole_graph_llm_assignment]

    llm_assignments = {node: whole_graph_llm_assignment[node] for node in G.nodes()}
    final_metrics = None

    ### EDGE CASE: only one node in the graph
    start_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes()))
    # print(f'start_nodes: {start_nodes}')
    # print(f'blending_reference {blending_node_db_references}')
    # print('start_nodes', start_nodes)
    # print('llm assignment')
    # pprint(llm_assignments)
    if len(start_nodes) == 1 and G.out_degree(start_nodes[0]) == 0:
        metrics = single_model_metrics[llm_assignments[start_nodes[0]]]
        return LLMMetrics(
            final_cost=metrics.final_cost,
            final_latency=metrics.final_latency,
            final_energy=metrics.final_energy,
            quality_of_answer=metrics.quality_of_answer
        )

    # print('s', start_nodes)

    for start_node in start_nodes:
        key = llm_assignments[start_node]
        metrics = copy.deepcopy(single_model_metrics[key])
        # zero out the final cost and final latency because we are calculating token-wise.
        # and the start nodes are still single models, so they won't be matched as a sub-schedule
        metrics.final_cost = 0
        metrics.final_latency = 0
        metrics.final_energy = 0
        queue.append(BFSNode(
            LLM_name=key,
            node=start_node,
            metrics=metrics,
            accumulated_metrics=metrics,
            first_node_in_seq=True,
        ))

    # print('initial state of the queue:')
    # pprint(queue)
    while queue:
        bfs_node = queue.popleft()
        LLM_name, node, metrics, current_metrics, first_node_in_seq = bfs_node

        # metrics = copy.deepcopy(metrics)
        # current_metrics = copy.deepcopy(current_metrics)
        # print('processing from queue', LLM_name)

        # might not need second part of AND condition
        if is_blend_node(G, node) and node not in matched_nodes_information:
            # Node has not received information from all its inputs, so reprocess at later time
            if len(processing_table[node]) != G.in_degree(node):
                queue.append(bfs_node)
                continue
            else:
                # calc_cost_parallel will return new reference of current_metrics, so original is not modified
                # print('p table for', node_key, node)
                # pprint(processing_table[node])
                current_metrics = calculate_cost_parallel(
                    llm_assignments,
                    node,
                    current_metrics,
                    metrics,
                    processing_table[node],
                    blending_node_db_references,
                    single_model_metrics,
                    query_tokens,
                    blending_prompt_tokens,
                    turn_off_blend_table_check=True,
                    G=G
                )

        successors = list(G.successors(node))
        if len(successors) == 0 and len(queue) == 0:
            final_metrics = current_metrics
            break

        # print(f'successors for {successors} for {llm_assignments[node]}')
        for successor in successors:
            # these keys are the LLM names based off their corresponding node integer
            node_key, successor_key = llm_assignments[node], llm_assignments[successor]
            # we came across a successor that is apart of a matched sub-schedule, but because it is in processed_sink, that means this successor has
            # already been added to the queue, hence seen, so we do not do anything here
            if successor in matched_nodes_information and successor in processed_sinks:
                continue

            # matched sub-schedule sink found, so we merely propagate and do not touch the associated metrics
            if successor in matched_nodes_information:
                processed_sinks.add(successor)
                metrics = matched_nodes_information[successor][
                    0]  # the 0 index is here because matched_node_information[key] points to list of info, 0th element has the metrics
                if isinstance(metrics, tuple):
                    metrics = convert_metrics_to_dict(metrics)
                # queue.append(BFSNode(LLM_name=successor_key, node=successor, metrics=metrics, accumulated_metrics=metrics, first_node_in_seq=False))
                queue.append(
                    BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key],
                            accumulated_metrics=metrics, first_node_in_seq=False))
                continue

            if is_blend_node(G, successor):
                # print(f'in here, current_node: {node_key}, successor: {successor_key} type node: {type(node_key)} type succ: {type(successor_key)}')
                # if we encounter a blend node, but the current node has no preds that means it is a single model
                # so we set the final_cost, latency, and energy to 0, because

                ncurrent_metrics = copy.deepcopy(current_metrics)
                # we do not use the calculate_cost_sequential_v2 here because the behavior of calculation is slightly different
                if is_start_node(G, node):
                    # print('in here for LLM', LLM_name)
                    cost = query_tokens * metrics.input_cost + metrics.average_output_tokens * metrics.output_cost
                    latency = query_tokens * metrics.input_latency + metrics.average_output_tokens * metrics.output_latency
                    energy = query_tokens * metrics.input_energy + metrics.average_output_tokens * metrics.output_energy
                    # print(metrics.input_cost, metrics.input_energy)
                    # print('node is first and succ is blend', cost, latency, energy)
                    ncurrent_metrics.final_cost = cost
                    ncurrent_metrics.final_latency = latency
                    ncurrent_metrics.final_energy = energy

                # if node is sequential node, and there are predecessors, then we assume that the cost incurred
                # by this node has been calculated previously when this node was a successor for something else
                # Also, if node is blend node, we can just propagate, because metric acclumation would have already happened in the top of the while loop
                processing_table[successor].append(ProcessingTableEntry(node=node, LLM=node_key, metrics=metrics,
                                                                        accumulated_metrics=ncurrent_metrics))
                # print('current state of processing table', processing_table[successor])
                # first time reaching blend node, so we will add it to the queue for processing, remember at the top of the while loop, we will readd the same blend node if
                # we do not see the required amount of information in the processing table
                if successor not in processed_sinks:
                    processed_sinks.add(successor)
                    # if successor_key != 5 :
                    queue.append(
                        BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key],
                                accumulated_metrics=single_model_metrics[successor_key], first_node_in_seq=False))

            # Case where successor is sequential, we handle metric aggregation different based on if it is the first node in the sequential chain
            else:
                # Case where it is the first node. In this case, we will add the costs incurred by A and B, also we know that node is not a blend node
                if first_node_in_seq:
                    ncurrent_metrics = copy.deepcopy(current_metrics)

                    cost, latency, energy = calculate_cost_sequential_v2(
                        query_tokens,
                        metrics.average_output_tokens,
                        ctx_prompt_tokens,
                        single_model_metrics[successor_key].average_output_tokens,
                        metrics,
                        single_model_metrics[successor_key],
                        True
                    )

                    ncurrent_metrics.final_cost = cost
                    ncurrent_metrics.final_latency = latency
                    ncurrent_metrics.final_energy = energy

                    queue.append(
                        BFSNode(
                            LLM_name=successor_key,
                            node=successor,
                            metrics=single_model_metrics[successor_key],
                            accumulated_metrics=ncurrent_metrics,
                            first_node_in_seq=False
                        )
                    )

                else:
                    successor_metrics = single_model_metrics[successor_key]
                    ncurrent_metrics = copy.deepcopy(current_metrics)
                    # if is_blend_node(G, node):
                    #   # qoa_estimation = (current_metrics.quality_of_answer + successor_metrics.quality_of_answer) / 2
                    #   # node -> successor
                    #   # def sequential_delta(node_A: str, node_B:str, single_model_metrics, pairwise_metrics):
                    #   #   final = pairwise_metrics[(node_A, node_B)].quality_of_answer
                    #   #   initial = single_model_metrics[node_A].quality_of_answer
                    #   #   delta = (final - initial) - initial
                    #   #   return delta
                    #   N = 0
                    #   delta = 0
                    #   for llm in llm_assignments:
                    #     if G.in_degree(llm) >= 2:
                    #       continue
                    #     N += 1
                    #     delta += sequential_delta(llm_assignments[llm], llm_assignments[successor], single_model_metrics, pairwise_metrics)
                    #   qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + (delta / N))
                    # else:
                    #   qoa_estimation = ncurrent_metrics.quality_of_answer * (1 + sequential_delta(llm_assignments[node], llm_assignments[successor], single_model_metrics, pairwise_metrics))

                    cost, latency, energy = calculate_cost_sequential_v2(
                        query_tokens,
                        single_model_metrics[node_key].average_output_tokens,
                        ctx_prompt_tokens,
                        successor_metrics.average_output_tokens,
                        None,
                        successor_metrics,
                        False,
                    )
                    # add to queue after aggregation
                    ncurrent_metrics.final_cost += cost
                    ncurrent_metrics.final_latency += latency
                    ncurrent_metrics.final_energy += energy
                    # ncurrent_metrics.quality_of_answer = qoa_estimation
                    queue.append(
                        BFSNode(LLM_name=successor_key, node=successor, metrics=single_model_metrics[successor_key],
                                accumulated_metrics=ncurrent_metrics, first_node_in_seq=False))

            # print('sadasd')
            # pprint(queue)

        # print('state of queue')
        # pprint(queue)

    return final_metrics


def get_sub_llm_assignment(subDAG, llm_assignment):
    return [llm_assignment[node] for node in sorted(subDAG.nodes())]


def get_seq_chains_starting_from(g, node, chains, path=None):
    # print('current path', path, 'current node', node, 'successor for this node', g.successors(node))
    if is_blend_node(g, node):
        chains.append(path)
        return

    if path is None:
        path = []

    if g.out_degree(node) == 0:
        chains.append(path + [node])
        return

    path = path + [node]  # create a new path to avoid mutation
    for successor in g.successors(node):
        get_seq_chains_starting_from(g, successor, chains, path)


def flatten_intermediate_blends_stagewise(G):
    """
    Remove only blending nodes (indegree>1) that feed into other blending nodes.
    For each such node, reattach its predecessors directly to its successors, then remove it.
    Repeat until no blend->blend edges remain.
    """
    H = G.copy()
    while True:
        # find blend nodes feeding other blend nodes
        blend_nodes = [
            n for n in H.nodes()
            if H.in_degree(n) > 1 and any(H.in_degree(s) > 1 for s in H.successors(n))
        ]
        if not blend_nodes:
            break
        n = blend_nodes[0]
        preds = list(H.predecessors(n))
        succs = [s for s in H.successors(n) if H.in_degree(s) > 1]
        # reattach preds -> those blend successors
        for p in preds:
            for s in succs:
                if p != s and not H.has_edge(p, s):
                    H.add_edge(p, s)
        # remove n's blend->blend connections and node
        H.remove_node(n)
    return H


def flatten_intermediate_blends_stagewise_mut(H):
    """
    Remove only blending nodes (indegree>1) that feed into other blending nodes.
    For each such node, reattach its predecessors directly to its successors, then remove it.
    Repeat until no blend -> blend edges remain.
    """
    while True:
        # find blend nodes feeding other blend nodes
        blend_nodes = [
            n for n in H.nodes()
            if H.in_degree(n) > 1 and any(H.in_degree(s) > 1 for s in H.successors(n))
        ]
        if not blend_nodes:
            break
        n = blend_nodes[0]
        preds = list(H.predecessors(n))
        succs = [s for s in H.successors(n) if H.in_degree(s) > 1]
        # reattach preds -> those blend successors
        for p in preds:
            for s in succs:
                if p != s and not H.has_edge(p, s):
                    H.add_edge(p, s)
        # remove n's blend->blend connections and node
        H.remove_node(n)


## NEW
"""
V3 matches starting sequential chains from the back of the chain
- after sequential chain matching, we fetch reference models for the blending nodes for parallel estimation
- using matches and the parallel strategy, we estimate any unmatched sequential components with the pairwise method
"""


def match_subDAGs_v3(
        G: nx.DiGraph,
        assignment: List[str],
        query_type: str,
        df_history: pd.DataFrame,
        single_model_metrics,
        pairwise_model_metrics,
        levenshtein_threshold: float = 0.75,
        turn_off_exact_fuzzy_matching=False,
        print_debug_log=True
):
    import collections
    debug_log = []
    matched_nodes_information = collections.defaultdict(list)
    # Get the final LLM/node in DAG to start our DFS from
    sink = list(filter(lambda node: G.out_degree(node) == 0, G.nodes()))[0]
    # construct the DAG behind sink
    subDAG, sub_llm_assignment, sink = construct_subdag_behind_node(G, sink, assignment)
    target_structure_id = canonical_label_for_isomorphism(subDAG)

    # First we fuzzy match the entire graph
    # check for exact match OR a very good fuzzy match

    # print('whole Graph', assignment)
    # plot_graph(G)
    if not turn_off_exact_fuzzy_matching:
        last_node_only = nx.DiGraph()
        last_node_only.add_node(sink)
        sink_node_llm = get_sub_llm_assignment(last_node_only, assignment)[0]
        best_candidate_llm_assignment, metrics, score = fuzzy_ranked_matches_from_database(
            sub_llm_assignment,
            target_structure_id,
            sink_node_llm,
            query_type,
            df_history,
            threshold=levenshtein_threshold,
        )

        # suitable match is found so let's store the metrics in our dictionary
        if metrics is not None and score >= levenshtein_threshold:
            # _metrics = (cost, latency, energy, qoa) : FORMAT
            cost, latency, energy, qoa = metrics
            matched_nodes_information[sink].append(
                LLMMetrics(final_cost=cost, final_latency=latency, final_energy=energy, quality_of_answer=qoa))
            matched_nodes_information[sink].append('fuzzy-match')
            matched_nodes_information[sink].append(best_candidate_llm_assignment)
            matched_nodes_information[sink].append(score)
            return metrics, {}, False  # False here tells the estimation function no need for traversal because we found a good exact OR fuzzy match on whole graph

        start_nodes = list(filter(lambda node: G.in_degree(node) == 0, G.nodes()))
        # print('start_nodes', start_nodes)
        sequential_chains = []
        for node in start_nodes:
            get_seq_chains_starting_from(G, node, sequential_chains)

        # the sequential chains [[[0, 1]], [[2, 3, 4]], [[6]]]
        # print(f'Sequential chains: {sequential_chains}')
        for seq_chain in sequential_chains:
            # ignore single models
            if len(seq_chain) == 1:
                continue
            # Temp Graph that we can afford to mutate structure
            TG = nx.DiGraph()
            TG.add_nodes_from(seq_chain)
            for i in range(1, len(seq_chain)):
                TG.add_edge(seq_chain[i - 1], seq_chain[i])

            # issue here, when using different start node in chain function, the ordering don't match up
            TG_llm_assignment = get_sub_llm_assignment(TG, assignment)
            # print('Graph of seq chain going into fuzzy', seq_chain, TG_llm_assignment)
            # plot_graph(TG)
            metrics, matchedG, last_node_in_sequential = fuzzy_match_sequential_chain(TG, TG_llm_assignment, query_type,
                                                                                      single_model_metrics, df_history,
                                                                                      levenshtein_threshold)
            # print('last node in sequential', last_node_in_sequential)
            if metrics is not None:
                matched_nodes_information[last_node_in_sequential].append(metrics)
                matched_nodes_information[last_node_in_sequential].append('sequential-match')

    # pprint(matched_nodes_information)
    # print('jj')
    # at this point fuzzy matching the entire graph is done, so we now merge the blend node dependencies
    flatten_intermediate_blends_stagewise_mut(G)
    # new llm assignment after removing the blend nodes
    # assignment = get_sub_llm_assignment_given_nodes(G.nodes(), assignment)
    # plot_graph(G)
    # print('llm assignment after flatten func', assignment)

    # here we go through each blend node, and gather its inputs, because of the flatten function call
    # if a blend nodes input was a blend node, the original blend node gets that blend nodes input
    # we look for matches on the blend_node and its input in the DB and store the qoa along with the qoa
    # of the input models
    blend_nodes = filter(lambda node: G.in_degree(node) >= 2, list(nx.topological_sort(G)))
    # print('blend_nodes', blend_nodes)
    # print('plotting graph after flattening jizzz')
    # plot_graph(G)
    # print('graph plotted')
    blending_node_db_references = {}
    for blend_node in blend_nodes:
        # print('in the blend loop')

        # create the dummy dag for searching
        inputs = list(G.predecessors(blend_node))
        llm_to_single_model_qoa_for_reference = {}
        dummy_DAG = nx.DiGraph()
        dummy_DAG.add_nodes_from(inputs + [blend_node])

        # print('nodes added to dummy dag', list(dummy_DAG.nodes()))

        for input in inputs:
            dummy_DAG.add_edge(input, blend_node)
            the_llm = get_sub_llm_assignment_given_nodes([input], assignment)[0]
            llm_to_single_model_qoa_for_reference[input] = single_model_metrics[the_llm]

        # get the llm_assignment for searching
        dummy_DAG_llm_assignment = get_sub_llm_assignment(dummy_DAG, assignment)
        # print('dummy lm', dummy_DAG_llm_assignment)
        # print('dummy_DAG_llm_assignment', dummy_DAG_llm_assignment)
        # plot_graph(dummy_DAG)

        # search
        # bn_reference_metrics = get_subdag_metrics_v7(
        #   dummy_DAG,
        #   dummy_DAG_llm_assignment,
        #   query_type,
        #   df_history
        # )

        bn_reference_metrics = special_get_subdag_metrics_for_one_blend_operations(
            dummy_DAG,
            dummy_DAG_llm_assignment,
            query_type,
            df_history
        )

        # based on our level scheme we are guaranteed to have them
        # print('blend graph')
        # plot_graph(dummy_DAG)
        # print('dummy dag nodes', list(dummy_DAG.nodes()))
        # for edge in dummy_DAG.edges:
        #   print('edge', edge)
        # print('blend graph llm assignment', dummy_DAG_llm_assignment)
        # print('original graph llm assignment', assignment)

        # UNIQUE COMMENT FOR BHARG, error happens because there is no match in DB for our parallel subgraph
        if bn_reference_metrics:
            blending_node_db_references[blend_node] = BlendingNodeDBReference(
                output=bn_reference_metrics.quality_of_answer,
                inputs=llm_to_single_model_qoa_for_reference
            )
        else:
            print("no bn_ref metrics found for", blend_node)

    return matched_nodes_information, blending_node_db_references, True  # True here indicates we need to perform the traversal for stats


def estimate_schedule_v3(
        G: nx.DiGraph,
        assignment: List[str],
        query_type: str,
        query_tokens: int,
        blending_prompt_tokens: int,
        ctx_tokens: int,
        df_history: pd.DataFrame,
        levenshtein_threshold: float = 0.75,
        turn_off_exact_fuzzy_matching=False
):
    # this function will return the last blend node and its inputs, if graph is full sequential then the returned dictionary will only contain content if there is a good fuzzy/exact match for
    # the full sequential graph
    original_graph_copy = G.copy()
    single_model_metrics = get_single_model_metrics(assignment, query_type, df_history)
    pairwise_metrics = get_pairwise_metrics(assignment, query_type, df_history)

    if len(single_model_metrics) == 0:
        # level zero return default stats
        return LLMMetrics(
            final_cost=0.5,
            final_latency=20,
            final_energy=20,
            quality_of_answer=0.5
        )
    elif len(pairwise_metrics) == 0:
        # Only single-model data available (e.g. level 0 cold start).
        # Use existing traversal with empty pairwise/match/blend data so that
        # per-token input/output costs and the QoA estimation functions are reused.
        matched_subDAG_information = {}
        blending_node_db_references = {}

        final_metrics = new_traversal_v3(
            G,
            pairwise_metrics,
            single_model_metrics,
            matched_subDAG_information,
            blending_node_db_references,
            query_tokens,
            ctx_tokens,
            blending_prompt_tokens,
            assignment,
        )
        if final_metrics is None:
            return LLMMetrics(final_cost=0.5, final_latency=20, final_energy=20, quality_of_answer=0.5)

        final_metrics_for_cost = new_traversal_just_for_cost_latency_energy_v3(
            original_graph_copy,
            pairwise_metrics,
            single_model_metrics,
            matched_subDAG_information,
            blending_node_db_references,
            query_tokens,
            ctx_tokens,
            blending_prompt_tokens,
            assignment,
        )
        final_metrics.final_cost = final_metrics_for_cost.final_cost
        final_metrics.final_energy = final_metrics_for_cost.final_energy
        final_metrics.final_latency = final_metrics_for_cost.final_latency
        return final_metrics

    # level 2 and above
    # first this function will fuzzy/exact match the entire graph, if cant be done, then fuzzy/exact match sequential chains from the start
    # then it gathers information on the blending node and their input's QoA, storing them in blending_node_db_references
    matched_subDAG_information, blending_node_db_references, need_traversal = match_subDAGs_v3(G, assignment,
                                                                                               query_type, df_history,
                                                                                               single_model_metrics,
                                                                                               pairwise_metrics,
                                                                                               levenshtein_threshold,
                                                                                               turn_off_exact_fuzzy_matching=turn_off_exact_fuzzy_matching)
    # pprint(matched_subDAG_information)
    # pprint(matched_subDAG_information)
    if not need_traversal:
        # in this case matched_subDAG_information is actually a tuple if need_traversal is False
        cost, latency, energy, qoa = matched_subDAG_information
        # info = list(matched_subDAG_information.keys())[0]
        # cost = info["final_cost"]
        # energy = info["final_energy"]
        # latency = info["final_latency"]
        # qoa = info["quality_of_answer"]
        # print("here 3", cost, latency, energy, qoa)
        return LLMMetrics(
            final_cost=cost,
            final_latency=latency,
            final_energy=energy,
            quality_of_answer=qoa
        )

    # print('made here, starting traversal')
    # plot_graph(G)
    # print('og', assignment)
    # traverse the graph, with information on matches and blending nodes for qoa estimation
    # plot_graph(G)
    # print(assignment)
    # print('nodes', list(G.nodes()))
    # for edge in G.edges:
    #   print('edge', edge)
    # print('assignment before going to traversal', assignment)
    # plot_graph(G)
    # print('matched_subdag_info')
    # pprint(matched_subDAG_information)
    final_metrics = new_traversal_v3(
        G,
        pairwise_metrics,
        single_model_metrics,
        matched_subDAG_information,
        blending_node_db_references,
        query_tokens,
        ctx_tokens,
        blending_prompt_tokens,
        assignment,
    )
    # ('returning final metrics after traversal')
    if final_metrics is None: raise RuntimeError(
        "new_traversal_v3 returned None: traversal never reached a terminal sink")

    # print('blending_node_db_ref before new_trav')
    # pprint(blending_node_db_references)

    final_metrics_for_just_cost_energy_latency = new_traversal_just_for_cost_latency_energy_v3(
        original_graph_copy,
        pairwise_metrics,
        single_model_metrics,
        matched_subDAG_information,
        blending_node_db_references,
        query_tokens,
        ctx_tokens,
        blending_prompt_tokens,
        assignment,
    )
    final_metrics.final_cost = final_metrics_for_just_cost_energy_latency.final_cost
    final_metrics.final_energy = final_metrics_for_just_cost_energy_latency.final_energy
    final_metrics.final_latency = final_metrics_for_just_cost_energy_latency.final_latency
    # print('all the shit for schedule is done')
    return final_metrics


class Individual:
    def __init__(self, struct_id, assignment):
        self.struct_id = struct_id  # DAG structure identifier (canonical mask)
        self.assignment = tuple(assignment)  # tuple of LLM model indices per node
        self.metrics = None  # (cost, latency, energy, qoa)
        self.objectives = None  # (cost, latency, energy, -qoa)
        self.rank = None
        self.crowding_distance = None


# Helper to get indegree list from structure
def get_indegrees(k, struct_id):
    indeg = [0] * k
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            # print("Left shift for", i, ",", j, "is", (1 << bit_index))
            if struct_id & (1 << bit_index):
                indeg[j] += 1
            bit_index += 1
    return indeg


# Initialize population with random DAGs and valid assignments
def initialize_population(pop_size, max_nodes):
    """
    Initialize population with random DAG structures and assignments.

    Uses random structure generation (no enumeration) for large max_nodes.
    For max_nodes <= 6, uses enumeration for better coverage.
    """
    population = []

    # For small max_nodes, use enumeration (fast and complete)
    # For large max_nodes, use random generation (avoids expensive enumeration)
    use_enumeration = (max_nodes <= 6)

    if use_enumeration:
        # Original approach: enumerate all structures
        structures = generate_nonisomorphic_dags(max_nodes)

        for _ in range(pop_size):
            k, struct_id = random.choice(structures)
            indeg = get_indegrees(k, struct_id)

            # Randomly assign base models, set blending for indegree>1
            assignment = [None] * k
            for i in range(k):
                assignment[i] = 5 if indeg[i] > 1 else random.randint(0, 4)

            individual = Individual(struct_id, assignment)
            population.append(individual)
    else:
        # Random generation approach: no enumeration needed
        print(f"  Using random structure generation (max_nodes={max_nodes} > 6)")
        print(f"  Note: Initialization may take 30-60s due to canonicalization")

        for i in range(pop_size):
            if i > 0 and i % 50 == 0:
                print(f"    Generated {i}/{pop_size} individuals...")

            # Randomly pick k value (uniform distribution)
            k = random.randint(1, max_nodes)

            # Generate random valid structure for this k
            # MUST canonicalize for PerfDB lookups to work correctly
            k, struct_id = random_valid_structure(k, skip_canonical=False)
            indeg = get_indegrees(k, struct_id)

            # Randomly assign base models, set blending for indegree>1
            assignment = [None] * k
            for i in range(k):
                assignment[i] = 5 if indeg[i] > 1 else random.randint(0, 4)

            individual = Individual(struct_id, assignment)
            population.append(individual)

    return population


# Crossover operator
def crossover(parent1, parent2):
    if len(parent1.assignment) != len(parent2.assignment):
        return copy.deepcopy(parent1 if random.random() < 0.5 else parent2)

    k = len(parent1.assignment)
    num_bits = k * (k - 1) // 2

    mask1, mask2 = int(parent1.struct_id), int(parent2.struct_id)

    if num_bits > 1:
        cp = random.randint(1, num_bits - 1)
        new_mask = ((mask1 & ((1 << cp) - 1)) | (mask2 & ~((1 << cp) - 1)))
    else:
        new_mask = mask1 if random.random() < 0.5 else mask2

    assign1, assign2 = list(parent1.assignment), list(parent2.assignment)
    if k > 1:
        cp2 = random.randint(1, k - 1)
        new_assign = assign1[:cp2] + assign2[cp2:]
    else:
        new_assign = assign1[:]

    # Repair: ensure only last node is sink => all nodes < k-1 have outdeg > 0
    outdeg = [0] * k
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if new_mask & (1 << bit_index):
                outdeg[i] += 1
            bit_index += 1

    for node in range(k - 1):
        if outdeg[node] == 0:
            idx = edge_bit_index(node, k - 1, k)
            new_mask |= (1 << idx)

    # Repair assignments for indegree constraints on the *current* mask
    indeg = get_indegrees(k, new_mask)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)

    # Canonicalize AND permute assignment using the best permutation
    adj = mask_to_adj(k, new_mask)
    canon_mask, perm = canonical_mask_and_perm(adj, k)

    if perm is not None:
        # new_pos -> old_node mapping in perm
        new_assign = [new_assign[old_node] for old_node in perm]

    # Enforce blend constraints after canonicalization (since indegrees can shift)
    indeg = get_indegrees(k, canon_mask)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)

    return Individual(canon_mask, new_assign)




def mutate(ind, max_nodes):
    k = len(ind.assignment)
    struct_id = int(ind.struct_id)
    assign = list(ind.assignment)

    # Add node mutation
    if k < max_nodes and random.random() < ADD_NODE_PROB:
        new_k = k + 1
        adj = mask_to_adj(k, struct_id)
        # expand
        adj2 = [[0]*new_k for _ in range(new_k)]
        for i in range(k):
            for j in range(k):
                adj2[i][j] = adj[i][j]
        # connect old sink to new sink
        adj2[k-1][k] = 1

        assign.append(random.randint(0, 4))

        canon_mask, perm = canonical_mask_and_perm(adj2, new_k)
        if perm is not None:
            assign = [assign[old_node] for old_node in perm]

        indeg = get_indegrees(new_k, canon_mask)
        for i in range(new_k):
            if indeg[i] > 1:
                assign[i] = 5
            elif indeg[i] <= 1 and assign[i] == 5:
                assign[i] = random.randint(0, 4)

        return Individual(canon_mask, assign)

    # Flip an edge
    if k > 1 and random.random() < FLIP_EDGE_PROB:
        num_bits = k * (k - 1) // 2
        bit_to_flip = random.randrange(num_bits)
        struct_id ^= (1 << bit_to_flip)

    # Mutate base model nodes
    for i in range(k):
        if assign[i] != 5 and random.random() < MODEL_MUTATION_PROB:
            assign[i] = random.randint(0, 4)

    # Repair: ensure only last node is sink
    outdeg = [0] * k
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if struct_id & (1 << bit_index):
                outdeg[i] += 1
            bit_index += 1

    for node in range(k - 1):
        if outdeg[node] == 0:
            idx = edge_bit_index(node, k - 1, k)
            struct_id |= (1 << idx)

    # Canonicalize + permute assignment
    adj = mask_to_adj(k, struct_id)
    canon_mask, perm = canonical_mask_and_perm(adj, k)
    if perm is not None:
        assign = [assign[old_node] for old_node in perm]

    # Enforce indegree constraints after canonicalization
    indeg = get_indegrees(k, canon_mask)
    for i in range(k):
        if indeg[i] > 1:
            assign[i] = 5
        elif indeg[i] <= 1 and assign[i] == 5:
            assign[i] = random.randint(0, 4)

    return Individual(canon_mask, assign)



# Non-dominated sorting and crowding distance (NSGA-II selection mechanisms)
def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p.dom_count = 0
        p.dominated_set = []
    for p in population:
        for q in population:
            # Check domination (p dominates q?)
            if (p.objectives[0] <= q.objectives[0] and
                p.objectives[1] <= q.objectives[1] and
                p.objectives[2] <= q.objectives[2] and
                p.objectives[3] <= q.objectives[3]) and \
                    (p.objectives != q.objectives):
                # p is at least as good in all objectives
                if (p.objectives[0] < q.objectives[0] or
                        p.objectives[1] < q.objectives[1] or
                        p.objectives[2] < q.objectives[2] or
                        p.objectives[3] < q.objectives[3]):
                    p.dominated_set.append(q)
            if (q.objectives[0] <= p.objectives[0] and
                q.objectives[1] <= p.objectives[1] and
                q.objectives[2] <= p.objectives[2] and
                q.objectives[3] <= p.objectives[3]) and \
                    (q.objectives[0] < p.objectives[0] or
                     q.objectives[1] < p.objectives[1] or
                     q.objectives[2] < p.objectives[2] or
                     q.objectives[3] < p.objectives[3]):
                p.dom_count += 1
        if p.dom_count == 0:
            p.rank = 0
            fronts[0].append(p)
    # Subsequent fronts
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_set:
                q.dom_count -= 1
                if q.dom_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # remove last empty front
    return fronts


def assign_crowding_distance(front):
    if not front:
        return
    n_obj = 4
    # Initialize
    for ind in front:
        ind.crowding_distance = 0
    # For each objective, sort and assign crowding distances
    for m in range(n_obj):
        front.sort(key=lambda x: x.objectives[m])
        # Extreme boundary points
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        if front[0].objectives[m] == front[-1].objectives[m]:
            continue
        # Normalize and accumulate distance for intermediate points
        for j in range(1, len(front) - 1):
            dist = (front[j + 1].objectives[m] - front[j - 1].objectives[m]) / \
                   (front[-1].objectives[m] - front[0].objectives[m] + 1e-9)
            front[j].crowding_distance += dist


def deduplicate_population(pop):
    """
    Remove duplicate solutions from population, keeping only unique ones.

    For discrete optimization problems (like DAG structure optimization),
    duplicates waste population slots and reduce diversity.

    When duplicates exist, keeps the one with better crowding distance.
    This preserves diversity pressure from NSGA-II.

    Args:
        pop: Population list of Individuals

    Returns:
        List of unique Individuals (no duplicates by struct_id + assignment)
    """
    seen = {}
    unique_pop = []

    for ind in pop:
        # Use (struct_id, assignment) as unique key
        key = (ind.struct_id, tuple(ind.assignment))

        if key not in seen:
            # First time seeing this solution
            seen[key] = ind
            unique_pop.append(ind)
        else:
            # Duplicate found - keep the one with better crowding distance
            existing = seen[key]
            if ind.crowding_distance > existing.crowding_distance:
                # Replace with better diversity metric
                idx = unique_pop.index(existing)
                unique_pop[idx] = ind
                seen[key] = ind

    return unique_pop


def tournament_selection(population):
    # Binary tournament: prefer lower rank, then higher crowding distance
    i, j = random.sample(range(len(population)), 2)
    a, b = population[i], population[j]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # ranks equal, use crowding distance
    return a if a.crowding_distance >= b.crowding_distance else b


def nsga2_optimize(query_tokens, blending_prompt_tokens, ctx_tokens, df_history,
                   pop_size=100, generations=75, max_nodes=5, query_type="Sports",
                   verbose=True):
    import math

    # <<< UPDATED: on error, return worst-case (cost, latency, energy, qoa)
    def safe_eval(individual, llm_assignment):
        t0 = time.perf_counter()
        try:
            return evaluate_individual_V2(
                individual.struct_id,
                llm_assignment,
                query_type,
                query_tokens,
                blending_prompt_tokens,
                ctx_tokens,
                df_history
            )
        except RuntimeError as e:
            print(f"[WARN] struct_id={individual.struct_id}, assignment={llm_assignment}: {e}")
            return math.inf, math.inf, math.inf, 0.0
        finally:
            dt = time.perf_counter() - t0
            if dt > 10:
                print(f"[SLOW] {dt:.1f}s struct_id={individual.struct_id} assignment={llm_assignment}")

    # 1) initialize
    pop = initialize_population(pop_size, max_nodes)

    # 2) evaluate initial pop
    pop_iter = tqdm(pop, desc="Evaluating", unit="ind") if verbose else pop
    for ind in pop_iter:
        llm_assignment = [str(x) for x in ind.assignment]
        c, t, e, q = safe_eval(ind, llm_assignment)
        ind.metrics = (c, t, e, q)
        ind.objectives = (c, t, e, -q)

    # 3) evolutionary loop
    gen_times = []
    total_start = time.time()

    gen_iter = tqdm(range(1, generations + 1), desc="time/generation", unit="gen") if verbose else range(1, generations + 1)
    for gen in gen_iter:
        gen_start = time.time()
        # nondominated sort & crowding
        fronts = fast_non_dominated_sort(pop)
        for f in fronts:
            assign_crowding_distance(f)

        # produce offspring
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = tournament_selection(pop), tournament_selection(pop)
            child = mutate(crossover(p1, p2), max_nodes)
            llm_assignment = [str(x) for x in child.assignment]
            c, t, e, q = safe_eval(child, llm_assignment)
            child.metrics = (c, t, e, q)
            child.objectives = (c, t, e, -q)
            offspring.append(child)

        # elitist selection
        combined = pop + offspring
        fronts = fast_non_dominated_sort(combined)
        new_pop = []
        f = 0
        while f < len(fronts) and len(new_pop) + len(fronts[f]) <= pop_size:
            assign_crowding_distance(fronts[f])
            new_pop.extend(fronts[f])
            f += 1
        if len(new_pop) < pop_size:
            assign_crowding_distance(fronts[f])
            fronts[f].sort(key=lambda x: x.crowding_distance, reverse=True)
            new_pop.extend(fronts[f][:pop_size - len(new_pop)])

        # Deduplicate population to improve diversity for discrete optimization
        new_pop = deduplicate_population(new_pop)

        # If deduplication reduced population size, fill with next front
        while len(new_pop) < pop_size and f + 1 < len(fronts):
            f += 1
            if len(new_pop) + len(fronts[f]) <= pop_size:
                assign_crowding_distance(fronts[f])
                new_pop.extend(fronts[f])
            else:
                assign_crowding_distance(fronts[f])
                fronts[f].sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = pop_size - len(new_pop)
                new_pop.extend(fronts[f][:needed])
                break

        pop = new_pop

        # Track generation time
        gen_elapsed = time.time() - gen_start
        gen_times.append(gen_elapsed)

    # Print final summary
    if verbose:
        total_elapsed = time.time() - total_start
        import psutil, os
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print("\n" + "=" * 80)
        print("NSGA-II COMPLETED")
        print("=" * 80)
        print(f"Total generations: {len(gen_times)}")
        print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
        print(f"Average time per generation: {sum(gen_times)/len(gen_times):.1f}s")
        print(f"Min generation time: {min(gen_times):.1f}s")
        print(f"Max generation time: {max(gen_times):.1f}s")
        print(f"Peak memory: {mem_mb:.0f}MB")
        print("=" * 80)

    # return Pareto front
    pareto_front = [ind for ind in pop if ind.rank == 0]
    return pareto_front

# ============================================================================
#                  FULLY OPTIMIZED INLINE OVERLAY (Single File)
# ============================================================================

# --- Safe tqdm stub (toggle DISABLE_TQDM) ---
DISABLE_TQDM = True
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


class _NoTQDM:
    def __init__(self, it=None, **kwargs):
        self.it = it

    def __iter__(self):
        return iter(self.it) if self.it is not None else iter(())

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb): return False

    def update(self, *a, **k): pass

    def set_description(self, *a, **k): pass

    def set_postfix(self, *a, **k): pass

    def close(self): pass


# Rebind tqdm used by earlier defs at runtime
tqdm = _NoTQDM if (DISABLE_TQDM or _tqdm is None) else _tqdm

# --- Shared helpers / flags ---
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
import math, contextlib, io

NSGA_DEBUG = True


def dprint(*a, flush=False, **k):
    if NSGA_DEBUG:
        print(*a, flush=flush, **k)


# ---------------- History normalization + indexing ----------------
_HISTORY_STATE: Dict[int, Dict[str, Any]] = {}


def _normalize_assign_str(s) -> str:
    if s is None: return ""
    if isinstance(s, float):
        try:
            if math.isnan(s): return ""
        except Exception:
            pass
    if isinstance(s, str) and s.strip().lower() in {"nan", "none"}: return ""
    s = str(s).strip().replace(" ", "")
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    while ",," in s: s = s.replace(",,", ",")
    if s.startswith(","): s = s[1:]
    if s.endswith(","): s = s[:-1]
    return s


def _build_history_index(df):
    import pandas as pd
    df = df.copy(deep=False)

    # normalize columns
    if "query_type" in df.columns:
        df["query_type"] = df["query_type"].astype(str).str.strip()
    else:
        df["query_type"] = ""

    if "llm_assignments" in df.columns:
        df["llm_assignments"] = df["llm_assignments"].apply(_normalize_assign_str)
    else:
        df["llm_assignments"] = ""

    df["llm_list"] = df["llm_assignments"].map(lambda s: [t for t in s.split(",") if t])
    df["llm_set"] = df["llm_list"].map(set)
    df["last_node"] = df["llm_list"].map(lambda xs: xs[-1] if xs else None)

    # --- 1) fast candidate slicing for fuzzy (your existing by_key) ---
    by_key = {}
    if {"structure_id", "query_type", "last_node"}.issubset(df.columns):
        for key, grp in df.groupby(["structure_id", "query_type", "last_node"], dropna=False):
            by_key[key] = grp

    # --- 2) exact O(1) lookups for get_subdag_metrics_v7 ---
    metric_cols = [
        "input_cost","input_latency","input_energy",
        "output_cost","output_latency","output_energy",
        "qoa","average_output_tokens",
        "cost","latency","energy",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    exact_means = (
        df.groupby(["structure_id", "query_type", "llm_assignments"], dropna=False)[metric_cols]
          .mean()
    )
    # exact_means is a DataFrame with MultiIndex keys:
    # (structure_id, query_type, llm_assignments) -> mean metrics

    # --- 3) your one-blend multiset index (existing) ---
    by_inner_multiset = {}
    for _, row in df.iterrows():
        sid = row.get("structure_id"); q = row.get("query_type")
        lst = row.get("llm_list") or []
        if not lst or sid is None or q is None:
            continue
        sink = lst[-1]
        inner = tuple(sorted(lst[:-1]))
        key = (sid, q, sink, inner)
        by_inner_multiset.setdefault(key, []).append(row)

    return {
        "df": df,
        "by_key": by_key,
        "by_inner_multiset": by_inner_multiset,
        "exact_means": exact_means,
    }



def _get_hist_state(df_history):
    key = id(df_history)
    st = _HISTORY_STATE.get(key)
    if st is None:
        dprint("[opt] building history index")
        st = _build_history_index(df_history)
        _HISTORY_STATE[key] = st
    return st


def clear_history_index(df_history=None):
    if df_history is None:
        _HISTORY_STATE.clear()
    else:
        _HISTORY_STATE.pop(id(df_history), None)


# ---------------- Token-level similarity helpers ----------------
FUZZY_THRESHOLD = 0.70


def _seq_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    try:
        from rapidfuzz.distance import Levenshtein as L
        return 1.0 - L.normalized_distance(a_tokens, b_tokens)
    except Exception:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a_tokens, b_tokens).ratio()


def _jaccard_set(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


# ---------------- Optimized fuzzy matching ----------------
def fuzzy_ranked_matches_from_database(
        llm_assignment: List[str],
        target_structure_id: int,
        target_last_node: Optional[str],
        query_type: str,
        df_history,
        threshold: float = FUZZY_THRESHOLD,
        w_seq: float = 0.5,
        w_jac: float = 0.5,
        bonus_for_last_node: float = 0.01
):
    st = _get_hist_state(df_history)
    df_norm = st["df"];
    by_key = st["by_key"]

    tgt_list = [t for t in (llm_assignment or []) if t]
    tgt_sink = target_last_node if target_last_node else (tgt_list[-1] if tgt_list else None)

    candidates = by_key.get((target_structure_id, query_type, tgt_sink))
    if candidates is None or len(candidates) == 0:
        subset = df_norm[(df_norm["structure_id"] == target_structure_id) & (df_norm["query_type"] == query_type)]
        if subset.empty: return (None, None, 0.0)
        candidates = subset

    best_list = None;
    best_metrics = None;
    best_score = 0.0
    for _, row in candidates.iterrows():
        cand_list = row["llm_list"]
        seq_sim = _seq_similarity(tgt_list, cand_list)
        jac_sim = _jaccard_set(tgt_list, cand_list)
        score = w_seq * seq_sim + w_jac * jac_sim
        if tgt_sink and cand_list and cand_list[-1] == tgt_sink: score += bonus_for_last_node
        if score > best_score:
            best_score = float(score)
            best_list = cand_list
            best_metrics = (row["cost"], row["latency"], row["energy"], row["qoa"])

    if best_list is not None and best_score >= threshold:
        return (best_list, best_metrics, best_score)
    return (None, None, best_score)


# ---------------- Exact / one-blend sub-DAG metrics ----------------
def get_subdag_metrics_v7(subdag, sub_assignment: List[str], query_type: str, df_history):
    adj_matrix = get_adj_from_graph(subdag)
    structure_id = canonical_representation(adj_matrix, len(sub_assignment))
    assignment_str = _normalize_assign_str(",".join([str(x) for x in sub_assignment]))

    st = _get_hist_state(df_history)
    exact = st["exact_means"]

    key = (structure_id, str(query_type).strip(), assignment_str)
    try:
        row = exact.loc[key]   # O(1) lookup
    except KeyError:
        return None

    def g(col, default=None):
        return float(row[col]) if col in row.index and pd.notna(row[col]) else default

    return LLMMetrics(
        input_cost=g("input_cost"),
        input_latency=g("input_latency"),
        input_energy=g("input_energy"),
        output_cost=g("output_cost"),
        output_latency=g("output_latency"),
        output_energy=g("output_energy"),
        quality_of_answer=g("qoa"),
        average_output_tokens=g("average_output_tokens"),
        final_cost=g("cost"),
        final_latency=g("latency"),
        final_energy=g("energy"),
    )



def special_get_subdag_metrics_for_one_blend_operations(subdag, sub_assignment: List[str], query_type: str, df_history):
    if not sub_assignment: return None
    adj_matrix = get_adj_from_graph(subdag)
    structure_id = canonical_representation(adj_matrix, len(sub_assignment))

    st = _get_hist_state(df_history);
    idx = st["by_inner_multiset"]
    sink = sub_assignment[-1];
    inner = tuple(sorted(sub_assignment[:-1]))
    key = (structure_id, query_type, sink, inner)
    rows = idx.get(key, []);
    if not rows: return None

    import pandas as pd
    dfp = pd.DataFrame(rows)
    return LLMMetrics(
        input_cost=dfp["input_cost"].mean() if "input_cost" in dfp else None,
        input_latency=dfp["input_latency"].mean() if "input_latency" in dfp else None,
        input_energy=dfp["input_energy"].mean() if "input_energy" in dfp else None,
        output_cost=dfp["output_cost"].mean() if "output_cost" in dfp else None,
        output_latency=dfp["output_latency"].mean() if "output_latency" in dfp else None,
        output_energy=dfp["output_energy"].mean() if "output_energy" in dfp else None,
        quality_of_answer=dfp["qoa"].mean() if "qoa" in dfp else None,
        average_output_tokens=dfp["average_output_tokens"].mean() if "average_output_tokens" in dfp else None,
        final_cost=dfp["cost"].mean() if "cost" in dfp else None,
        final_latency=dfp["latency"].mean() if "latency" in dfp else None,
        final_energy=dfp["energy"].mean() if "energy" in dfp else None,
    )


# ---------------- Memoized evaluation wrapper ----------------
try:
    _orig_evaluate_individual_V2 = evaluate_individual_V2

    @lru_cache(maxsize=200000)
    def _eval_cache(struct_id: int, assignment_tuple: tuple, query_type: str,
                    query_tokens: int, blend_tokens: int, ctx_tokens: int,
                    df_key: int, cache_bust_token: int):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # use normalized df stored in history state (same columns, normalized strings)
            df = _HISTORY_STATE[df_key]["df"]
            return _orig_evaluate_individual_V2(
                struct_id,
                list(assignment_tuple),
                query_type,
                int(query_tokens),
                int(blend_tokens),
                int(ctx_tokens),
                df
            )

    def evaluate_individual_V2(struct_id, assignment, query_type,
                               query_tokens, blending_prompt_tokens, ctx_tokens,
                               df_history, cache_bust_token: Optional[int] = 0):
        df_key = id(df_history)
        _get_hist_state(df_history)  # ensure indexed
        return _eval_cache(
            int(struct_id),
            tuple(assignment or []),
            str(query_type),
            int(query_tokens),
            int(blending_prompt_tokens),
            int(ctx_tokens),
            df_key,
            int(cache_bust_token or 0)
        )
except Exception:
    pass


# ---------------- Cached canonicalization + DAG gen ----------------
try:
    _orig_canonical_representation = canonical_representation


    def _adj_to_tuple(adj):
        return tuple(tuple(int(v) for v in row) for row in adj)


    @lru_cache(maxsize=20000)
    def _canonical_cached_from_tuple(adj_tup, k: int) -> int:
        adj = [list(row) for row in adj_tup]
        return _orig_canonical_representation(adj, k)


    def canonical_representation(adj_matrix, k: int) -> int:
        return _canonical_cached_from_tuple(_adj_to_tuple(adj_matrix), int(k))
except Exception:
    pass

try:
    _orig_generate_nonisomorphic_dags = generate_nonisomorphic_dags


    @lru_cache(maxsize=16)
    def generate_nonisomorphic_dags(max_nodes: int):
        dprint(f"[opt] generate_nonisomorphic_dags(k={max_nodes}) [cached]")
        return _orig_generate_nonisomorphic_dags(int(max_nodes))
except Exception:
    pass


# ---------------- Diversity utilities ----------------
def geno_key(ind) -> Tuple[int, Tuple[str, ...]]:
    return (int(ind.struct_id), tuple(ind.assignment))


def enforce_genotype_dedup(population):
    seen = set();
    new_pop = []
    for ind in population:
        k = geno_key(ind)
        if k not in seen:
            seen.add(k);
            new_pop.append(ind)
    return new_pop


def infer_model_pool_from_history(df_history) -> List[str]:
    st = _get_hist_state(df_history);
    df = st["df"]
    models = set()
    for lst in df["llm_list"]:
        models.update(lst)
    return sorted(models)


def inject_random_immigrants(population, n_new: int, structures, model_pool: List[str], IndividualClass=None):
    import random
    if n_new <= 0 or not structures or not model_pool: return population
    if IndividualClass is None: IndividualClass = globals().get("Individual", None)
    for _ in range(n_new):
        s = random.choice(structures)
        if isinstance(s, (tuple, list)) and len(s) == 2:
            num_nodes, struct_id = int(s[0]), int(s[1])
        else:
            struct_id = getattr(s, "struct_id", None) or getattr(s, "id", None)
            num_nodes = getattr(s, "num_nodes", None) or getattr(s, "k", None)
            if num_nodes is None:
                try:
                    num_nodes = len(s[0])
                except Exception:
                    num_nodes = 3
            if struct_id is None: continue
        assign = [random.choice(model_pool) for _ in range(int(num_nodes))]
        if IndividualClass is not None:
            ind = IndividualClass(struct_id=struct_id, assignment=assign)
        else:
            class _Tmp:
                __slots__ = ("struct_id", "assignment")

            def __init__(self, sid, asg):
                self.struct_id, self.assignment = sid, asg

            ind = _Tmp(struct_id, assign)
        population.append(ind)
    return population


class EarlyStopper:
    def __init__(self, patience: int = 8):
        self.patience = patience;
        self.stale = 0;
        self.last_keys = set()

    def update_and_should_stop(self, pareto_front) -> bool:
        keys = {(int(ind.struct_id), tuple(ind.assignment)) for ind in pareto_front}
        if keys.issubset(self.last_keys):
            self.stale += 1
        else:
            self.stale = 0; self.last_keys = keys
        return self.stale >= self.patience


def jitter_objectives(obj_tuple: Tuple[float, ...], eps: float = 1e-6) -> Tuple[float, ...]:
    import random
    return tuple(x + eps * random.random() for x in obj_tuple)


# =============================================================================
#                  RandomMOQO: Multi-Objective Query Optimization
# =============================================================================

# Global cache for plan evaluations
_evaluation_cache = {}


def clear_evaluation_caches():
    """Clear all evaluation caches so each algorithm starts with a cold cache."""
    global _evaluation_cache
    _evaluation_cache.clear()
    # Clear the history state index (keyed by DataFrame id)
    _HISTORY_STATE.clear()
    # Clear the fitness function lru_cache
    try:
        _eval_cache.cache_clear()
    except (NameError, AttributeError):
        pass
    # Clear canonical representation cache
    try:
        _canonical_cached_from_tuple.cache_clear()
    except (NameError, AttributeError):
        pass
    # Clear MOQO's own evaluation cache (separate dict in moqo.py)
    try:
        from src.llm_dag_optimizer.core import moqo as _moqo_mod
        _moqo_mod._evaluation_cache.clear()
    except (ImportError, AttributeError):
        pass
    # NOTE: do NOT clear generate_nonisomorphic_dags cache — DAG topology
    # enumeration is deterministic and independent of query data.

def prune_pareto(plans_with_metrics, alpha=1.0):
    """plans_with_metrics: List[(plan, metrics)]"""
    out = []
    for p, m in plans_with_metrics:
        out = Prune_Basic(out, p, m, alpha=alpha)
    return out

def node_type(G, node):
    indeg = G.in_degree(node)
    if indeg == 0: return "start"
    if indeg == 1: return "seq"
    return "blend"
def graph_from_struct(k: int, struct_id: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(k))
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if struct_id & (1 << bit_index):
                G.add_edge(i, j)
            bit_index += 1
    return G
# =============================================================================
# RandomMOQO Implementation (DEPRECATED - Use randommoqo.py module instead)
# =============================================================================
# NOTE: The implementations below have known bugs and are kept only for
# reference. The corrected implementations are in randommoqo.py module.
#
# Known issues in these implementations:
# 1. ParetoClimb doesn't update plan structure (only metrics)
# 2. ParetoClimbStep evaluates same assignment repeatedly
# 3. No adaptive precision in ApproximateFrontiers
#
# Use: from .randommoqo import RandomMOQO (imported at end of file)
# =============================================================================

def renumber_graph_nodes(G):
    """
    Renumber graph nodes to start from 0, maintaining edge structure.

    Returns:
        (new_graph, node_mapping) where node_mapping[old_node] = new_node
    """
    sorted_nodes = sorted(G.nodes())
    node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(sorted_nodes)}

    new_G = nx.DiGraph()
    new_G.add_nodes_from(range(len(sorted_nodes)))
    for u, v in G.edges():
        new_G.add_edge(node_mapping[u], node_mapping[v])

    return new_G, node_mapping


def ParetoClimbStep(plan: Individual, df_history, query_type, alpha=1.0):
    """
    Paper-style: recursively compute Pareto sets for subgraphs feeding each node,
    then combine child Pareto sets at that node to build parent candidates.
    Returns: List[(Individual, metrics)] non-dominated candidates for THIS subgraph/root.
    """
    k = len(plan.assignment)
    G = graph_from_struct(k, int(plan.struct_id))
    sink = [n for n in G.nodes() if G.out_degree(n) == 0][0]

    # Build subgraph behind sink (your existing helper is perfect)
    subG, sub_assign, sub_sink = construct_subdag_behind_node(G, sink, list(plan.assignment))

    # We'll do memoized recursion on nodes inside subG
    memo = {}

    def rec(node):
        """Return Pareto set of candidates for the subgraph ending at `node`.
        Each candidate is (assignment_variant, metrics_for_subdag, Individual_for_subdag_struct)
        We keep it simple: return List[(Individual, metrics)] for the subDAG behind node.
        """
        if node in memo:
            return memo[node]

        # Build the exact subDAG behind this node (preserves structure)
        local_subG, local_assign, _ = construct_subdag_behind_node(subG, node, sub_assign)

        preds = list(local_subG.predecessors(node))
        ntype = node_type(local_subG, node)

        # ---- Base case: start node (single model) ----
        if ntype == "start":
            # subDAG is just that node - use the assignment from construct_subdag_behind_node
            llm = local_assign  # list of 1

            # Renumber local_subG nodes to start from 0
            local_subG_renumbered, _ = renumber_graph_nodes(local_subG)

            metrics = get_subdag_metrics_v7(local_subG_renumbered, llm, query_type, df_history)
            if metrics is None:
                # fallback: use your DEFAULT_QOA scheme or inf costs
                metrics = LLMMetrics(final_cost=float("inf"), final_latency=float("inf"),
                                     final_energy=float("inf"), quality_of_answer=DEFAULT_QOA)
            # Represent candidate as an Individual on "some struct_id" is optional here,
            # since metrics are what you really use; keep a lightweight placeholder:
            cand = (("ASSIGN", tuple(llm)), metrics)
            memo[node] = [cand]
            return memo[node]

        # ---- Recursive case: sequential node (1 pred) ----
        if ntype == "seq":
            parent = preds[0]
            left_set = rec(parent)   # Pareto set feeding into parent

            results = []
            # Combine each parent-variant with node's model choice (operator mutation analog)
            for tag, m_parent in left_set:
                # We evaluate FULL subDAG behind node using the ACTUAL assignment
                # (this is simplest + faithful combo behavior).
                subDAG_node, llm_list, _ = construct_subdag_behind_node(local_subG, node, sub_assign)

                # Renumber subDAG nodes to start from 0 so evaluation functions work correctly
                subDAG_renumbered, _ = renumber_graph_nodes(subDAG_node)

                metrics = get_subdag_metrics_v7(subDAG_renumbered, llm_list, query_type, df_history)

                if metrics is None:
                    # allow fuzzy/estimation fallback if you want:
                    metrics = estimate_schedule_v3(
                        subDAG_renumbered, llm_list, query_type,
                        query_tokens=215, blending_prompt_tokens=26, ctx_tokens=39,
                        df_history=df_history,
                        levenshtein_threshold=0.75,
                        turn_off_exact_fuzzy_matching=False
                    )

                results.append((("ASSIGN", tuple(llm_list)), metrics))

            memo[node] = prune_pareto(results, alpha=alpha)
            return memo[node]

        # ---- Recursive case: blend node (>=2 preds) ----
        # Paper analog: combine leftPareto x rightPareto (generalize to many preds)
        child_sets = [rec(p) for p in preds]

        # Cartesian product over Pareto sets of each input
        results = []

        # Create node-to-assignment mapping for subG (since sub_assign is indexed by position, not node number)
        subG_nodes_sorted = sorted(subG.nodes())
        node_to_assign = {node: sub_assign[i] for i, node in enumerate(subG_nodes_sorted)}

        for combo in itertools.product(*child_sets):
            # Build the star-shaped subDAG with preds -> node (like your blending reference)
            # IMPORTANT: Renumber nodes to 0, 1, 2, ... so assignment indices match
            original_nodes = sorted(preds + [node])
            node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(original_nodes)}

            dummy = nx.DiGraph()
            dummy.add_nodes_from(range(len(original_nodes)))
            for p in preds:
                dummy.add_edge(node_mapping[p], node_mapping[node])

            # Extract assignments in the new node order
            llm_list = [node_to_assign[old_node] for old_node in original_nodes]

            # special handling for single-blend permutations (you already wrote this!)
            metrics = special_get_subdag_metrics_for_one_blend_operations(dummy, llm_list, query_type, df_history)
            if metrics is None:
                metrics = estimate_schedule_v3(
                    dummy, llm_list, query_type,
                    query_tokens=215, blending_prompt_tokens=26, ctx_tokens=39,
                    df_history=df_history,
                    levenshtein_threshold=0.75,
                    turn_off_exact_fuzzy_matching=False
                )
            results.append((("ASSIGN", tuple(llm_list)), metrics))

        memo[node] = prune_pareto(results, alpha=alpha)
        return memo[node]

    # Return the Pareto set for the whole plan (subDAG behind sink)
    pareto_for_whole = rec(sub_sink)
    return pareto_for_whole


def evaluate_plan_cached(plan, df_history, query_type):
    """Evaluate plan with caching"""
    cache_key = (plan.struct_id, tuple(plan.assignment), query_type)

    if cache_key in _evaluation_cache:
        return _evaluation_cache[cache_key]

    llm_assignment = [str(x) for x in plan.assignment]

    try:
        cost, latency, energy, qoa = evaluate_individual_V2(
            plan.struct_id,
            llm_assignment,
            query_type,
            query_tokens=215,
            blending_prompt_tokens=26,
            ctx_tokens=39,
            df_history=df_history
        )
        metrics = (cost, latency, energy, qoa)
        _evaluation_cache[cache_key] = metrics
        return metrics
    except Exception as e:
        import math
        return (math.inf, math.inf, math.inf, 0.0)


def dominates_strict(metrics1, metrics2, epsilon=1e-6):
    """Standard Pareto dominance (exact)"""
    # Handle None metrics (invalid/failed evaluations)
    if metrics1 is None or metrics2 is None:
        return False

    # Handle both tuple (cost, latency, energy, qoa) and LLMMetrics objects
    if isinstance(metrics1, LLMMetrics):
        c1, l1, e1, q1 = metrics1.final_cost, metrics1.final_latency, metrics1.final_energy, metrics1.quality_of_answer
    else:
        c1, l1, e1, q1 = metrics1

    if isinstance(metrics2, LLMMetrics):
        c2, l2, e2, q2 = metrics2.final_cost, metrics2.final_latency, metrics2.final_energy, metrics2.quality_of_answer
    else:
        c2, l2, e2, q2 = metrics2

    better_or_equal = (
        c1 <= c2 + epsilon and
        l1 <= l2 + epsilon and
        e1 <= e2 + epsilon and
        q1 >= q2 - epsilon
    )

    strictly_better = (
        c1 < c2 - epsilon or
        l1 < l2 - epsilon or
        e1 < e2 - epsilon or
        q1 > q2 + epsilon
    )

    return better_or_equal and strictly_better


def SigBetter(metrics1, metrics2, alpha):
    """
    Checks if metrics1 is significantly better than metrics2
    using coarsening factor α (α-dominance)
    """
    # Handle None metrics (invalid/failed evaluations)
    if metrics1 is None or metrics2 is None:
        return False

    # Handle both tuple (cost, latency, energy, qoa) and LLMMetrics objects
    if isinstance(metrics1, LLMMetrics):
        c1, l1, e1, q1 = metrics1.final_cost, metrics1.final_latency, metrics1.final_energy, metrics1.quality_of_answer
    else:
        c1, l1, e1, q1 = metrics1

    if isinstance(metrics2, LLMMetrics):
        c2, l2, e2, q2 = metrics2.final_cost, metrics2.final_latency, metrics2.final_energy, metrics2.quality_of_answer
    else:
        c2, l2, e2, q2 = metrics2

    # Apply alpha approximation (multiplicative slack)
    better_or_equal = (
        c1 <= alpha * c2 and
        l1 <= alpha * l2 and
        e1 <= alpha * e2 and
        q1 >= q2 / alpha
    )

    # Strictly better in at least one
    strictly_better = (
        c1 < c2 or
        l1 < l2 or
        e1 < e2 or
        q1 > q2
    )

    return better_or_equal and strictly_better


def Prune_Basic(plans, newPlan, newPlan_metrics, alpha=1.0):
    """
    Keeps Pareto optimal plans using alpha-approximate dominance

    Args:
        plans: List of (plan, metrics) tuples
        newPlan: New plan to consider
        newPlan_metrics: Metrics of new plan
        alpha: Approximation factor (1.0 = exact)

    Returns:
        Updated plans list
    """
    # Check if any existing plan approximately dominates new plan
    dominated_by_existing = False
    for p, p_metrics in plans:
        if SigBetter(p_metrics, newPlan_metrics, alpha):
            dominated_by_existing = True
            break

    if dominated_by_existing:
        return plans

    # Remove plans that are dominated by new plan
    filtered_plans = []
    for p, p_metrics in plans:
        if not SigBetter(newPlan_metrics, p_metrics, 1.0):
            filtered_plans.append((p, p_metrics))

    # Add new plan
    filtered_plans.append((newPlan, newPlan_metrics))

    return filtered_plans


def is_valid_dag_structure(struct_id, k):
    """Check if structure is valid DAG"""
    adj = mask_to_adj(k, struct_id)

    # Check single sink (only last node has out-degree 0)
    outdeg = [sum(adj[i]) for i in range(k)]
    if any(outdeg[i] == 0 for i in range(k-1)):
        return False
    if outdeg[k-1] != 0:
        return False

    return True


def repair_assignment_for_structure(struct_id, k, assignment):
    """Repair assignment to match structure's indegree constraints"""
    indeg = get_indegrees(k, struct_id)
    new_assignment = list(assignment)

    for i in range(k):
        if indeg[i] > 1 and new_assignment[i] != 5:
            new_assignment[i] = 5
        elif indeg[i] <= 1 and new_assignment[i] == 5:
            new_assignment[i] = random.randint(0, 4)

    return new_assignment


def generate_all_mutations(p):
    """
    Generate all single-step mutations of plan p

    Types of mutations:
    1. Change LLM assignment at one node
    2. Flip one edge in DAG
    """
    mutations = []
    k = len(p.assignment)

    # Mutation Type 1: Change LLM assignment
    indeg = get_indegrees(k, p.struct_id)
    for node_idx in range(k):
        if indeg[node_idx] > 1:
            continue  # Can't change blending nodes

        current_llm = p.assignment[node_idx]
        for new_llm in [0, 1, 2, 3, 4]:
            if new_llm != current_llm:
                new_assignment = list(p.assignment)
                new_assignment[node_idx] = new_llm
                mutations.append(Individual(p.struct_id, new_assignment))

    # Mutation Type 2: Flip one edge
    num_edges = k * (k - 1) // 2
    for bit in range(num_edges):
        new_struct_id = p.struct_id ^ (1 << bit)

        # Check if still valid DAG
        if is_valid_dag_structure(new_struct_id, k):
            # Repair assignment for new structure
            new_assignment = repair_assignment_for_structure(
                new_struct_id, k, list(p.assignment)
            )
            mutations.append(Individual(new_struct_id, new_assignment))

    return mutations


def ParetoStep(p, df_history, query_type):
    """
    Improve plan p by parallel local transformations
    Returns set of non-dominated mutations
    """
    pPareto = []

    # Generate all 1-step mutations
    mutations = generate_all_mutations(p)

    # Prune to keep only Pareto optimal mutations
    for mutated in mutations:
        mutated_metrics = evaluate_plan_cached(mutated, df_history, query_type)
        pPareto = Prune_Basic(pPareto, mutated, mutated_metrics, alpha=1.0)

    return pPareto


def ParetoClimb(p, df_history, query_type, max_iterations=50, alpha=1.0):
    improving = True
    iterations = 0

    # Evaluate current plan once
    p_metrics = evaluate_plan_cached(p, df_history, query_type)

    while improving and iterations < max_iterations:
        improving = False
        iterations += 1

        # PAPER-faithful step: get Pareto set for whole plan via recursion+combination
        candidates = ParetoClimbStep(p, df_history, query_type, alpha=alpha)

        # Pick any dominating candidate (paper uses “if dominates then move”)
        for _, cand_metrics in candidates:
            if dominates_strict(cand_metrics, p_metrics):
                # move to improved plan (keep structure/assignment as-is in this minimal version)
                # if you want to actually *change* p, you need to store a real mutated Individual per candidate
                # (see note below).
                p_metrics = cand_metrics
                improving = True
                break

    return p



def ApproximateFrontiers(p, P, i, query_type, df_history):
    """
    Algorithm 3: Approximates the Pareto frontier using adaptive precision

    Args:
        p: New optimized plan
        P: Plan cache dictionary
        i: Iteration count
        query_type: Query category
        df_history: Historical metrics

    Returns:
        Updated plan cache P
    """
    # Calculate target approximation precision
    # For this problem, use exact dominance from the start
    # Original paper used coarse-to-fine for very large search spaces
    # Our problem is smaller, so we can afford exact dominance
    alpha = 1.0  # Exact Pareto dominance

    if query_type not in P:
        P[query_type] = []

    # Evaluate new plan
    p_metrics = evaluate_plan_cached(p, df_history, query_type)

    # Prune with adaptive precision
    P[query_type] = Prune_Basic(P[query_type], p, p_metrics, alpha)

    return P


def RandomPlan(max_nodes):
    """
    Generate random bushy plan (random DAG structure + assignment).

    Uses random generation for max_nodes > 6 to avoid enumeration bottleneck.
    """
    # For small max_nodes, use enumeration (fast & complete)
    # For large max_nodes, use random generation (avoids O(k!) enumeration)
    if max_nodes <= 6:
        structures = generate_nonisomorphic_dags(max_nodes)
        k, struct_id = random.choice(structures)
    else:
        # Random generation: pick random k, then generate valid structure
        k = random.randint(1, max_nodes)
        k, struct_id = random_valid_structure(k, skip_canonical=False)

    indeg = get_indegrees(k, struct_id)
    assignment = []
    for i in range(k):
        if indeg[i] > 1:
            assignment.append(5)  # Blending node
        else:
            assignment.append(random.randint(0, 4))

    plan = Individual(struct_id, assignment)
    return plan


def RandomMOQO(query_type, df_history, timeout_seconds=60, max_nodes=5):
    """
    Algorithm 1: Main RandomMOQO algorithm
    Multi-objective optimization via random plans + local search

    Args:
        query_type: Query category (e.g., "Art")
        df_history: Historical metrics DataFrame
        timeout_seconds: Time limit for optimization
        max_nodes: Maximum nodes in DAG

    Returns:
        List of Pareto optimal plans
    """
    # Initialize partial plan cache and iteration counter
    P = {}
    P[query_type] = []
    i = 1

    start_time = time.time()
    print(f"\nRandomMOQO: Starting optimization for {query_type}")
    print(f"  Timeout: {timeout_seconds}s, Max nodes: {max_nodes}")

    # Refine frontier approximation until timeout
    while (time.time() - start_time) < timeout_seconds:
        # Generate random bushy query plan
        plan = RandomPlan(max_nodes)

        # Improve plan via fast local search
        optPlan = ParetoClimb(plan, df_history, query_type)

        # Approximate Pareto frontier with adaptive precision
        P = ApproximateFrontiers(optPlan, P, i, query_type, df_history)

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Iter {i}: {len(P[query_type])} solutions, {elapsed:.1f}s elapsed")

        i += 1

    elapsed = time.time() - start_time
    print(f"RandomMOQO: Completed {i} iterations in {elapsed:.1f}s")
    print(f"  Final Pareto set size: {len(P[query_type])}")

    # Extract just the plans (remove metrics from tuples)
    pareto_plans = [plan for plan, metrics in P[query_type]]

    # Evaluate all plans to set their metrics attribute
    for plan in pareto_plans:
        if plan.metrics is None:
            metrics = evaluate_plan_cached(plan, df_history, query_type)
            plan.metrics = metrics
            plan.objectives = (metrics[0], metrics[1], metrics[2], -metrics[3])

    return pareto_plans


# Import fixed RandomMOQO implementation from module
from .randommoqo import RandomMOQO


# =============================================================================
# Experiment Framework: Vary k and PerfDB Levels
# =============================================================================

def run_experiment(args, query_tokens, blending_prompt_tokens, ctx_tokens):
    """
    Run full experiment varying k (max_nodes) and PerfDB levels.

    For each query type:
    - Vary k from 1 to 5 (or specified range)
    - Test with PerfDB levels 0 to 4 (or specified range)
    - Run both NSGA-II and RandomMOQO
    - Save results with k and level metadata
    - Aggregate results to analyze impact of k and coverage
    """
    # Parse k values and levels
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    levels = [int(lv.strip()) for lv in args.levels.split(',')]

    print('=' * 80)
    print('EXPERIMENT: Varying k and PerfDB Coverage Levels')
    print('=' * 80)
    print(f'Query types: {QUERY_TYPES}')
    print(f'k values (max_nodes): {k_values}')
    print(f'PerfDB levels: {levels}')
    print(f'Total configurations: {len(k_values)} k × {len(levels)} levels = {len(k_values) * len(levels)}')
    print(f'Algorithm: {"Both" if args.algorithm == "both" else args.algorithm.upper()}')
    print('=' * 80)
    print()

    # Results storage
    all_results = []

    # Determine which algorithms to run
    run_nsga = args.algorithm in ["both", "nsga"]
    run_moqo = args.algorithm in ["both", "moqo"]

    # Main experiment loop
    config_num = 0
    total_configs = len(k_values) * len(levels)

    for level in levels:
        # Load data for this PerfDB level
        level_file = f"{args.level_dir}/level_{level}_data.csv"
        print(f"\n{'=' * 80}")
        print(f"Loading PerfDB Level {level}: {level_file}")
        print(f"{'=' * 80}")

        try:
            df_history = pd.read_csv(level_file)
            df_history['llm_assignments'] = (
                df_history['llm_assignments']
                .astype(str)
                .str.replace(r'[\(\)\s]', '', regex=True)
                .str.rstrip(',')
            )
            print(f"✓ Loaded {len(df_history)} records")
        except FileNotFoundError:
            print(f"✗ File not found: {level_file}")
            continue

        for k in k_values:
            config_num += 1
            print(f"\n{'-' * 80}")
            print(f"Configuration {config_num}/{total_configs}: k={k}, Level={level}")
            print(f"{'-' * 80}")

            for qt in QUERY_TYPES:
                print(f"\n  Query: {qt}")

                # Clear evaluation cache
                _evaluation_cache.clear()

                nsga_time = None
                nsga_solutions = 0
                moqo_time = None
                moqo_solutions = 0

                # ===== Run NSGA-II =====
                if run_nsga:
                    print(f"    [NSGA-II] Starting...")
                    start = time.time()

                    nsga_pareto = nsga2_optimize(
                        query_tokens,
                        blending_prompt_tokens,
                        ctx_tokens,
                        df_history,
                        pop_size=POP_SIZE,
                        generations=GENERATIONS,
                        max_nodes=k,  # Use current k value
                        query_type=qt
                    )

                    nsga_time = time.time() - start
                    nsga_solutions = len(nsga_pareto)

                    print(f"    [NSGA-II] Completed: {nsga_solutions} solutions in {nsga_time:.1f}s")

                    # Save NSGA-II results
                    for idx, ind in enumerate(nsga_pareto):
                        all_results.append({
                            'algorithm': 'NSGA-II',
                            'k': k,
                            'level': level,
                            'query_type': qt,
                            'solution_id': idx,
                            'struct_id': ind.struct_id,
                            'assignment': str(ind.assignment),
                            'cost': ind.objectives[0] if ind.objectives else None,
                            'energy': ind.objectives[1] if ind.objectives else None,
                            'latency': ind.objectives[2] if ind.objectives else None,
                            'qoa': -ind.objectives[3] if ind.objectives else None,
                            'runtime': nsga_time
                        })

                # ===== Run RandomMOQO =====
                if run_moqo:
                    _evaluation_cache.clear()

                    # Determine timeout
                    if args.moqo_time:
                        moqo_timeout = args.moqo_time
                    elif nsga_time:
                        moqo_timeout = int(nsga_time)
                    else:
                        moqo_timeout = 60

                    print(f"    [RandomMOQO] Starting (timeout: {moqo_timeout}s)...")
                    start = time.time()

                    moqo_pareto = RandomMOQO(
                        query_type=qt,
                        df_history=df_history,
                        timeout_seconds=moqo_timeout,
                        max_nodes=k,  # Use current k value
                        use_adaptive_precision=True,
                        alpha_start=2.0,
                        alpha_end=1.0
                    )

                    moqo_time = time.time() - start
                    moqo_solutions = len(moqo_pareto)

                    print(f"    [RandomMOQO] Completed: {moqo_solutions} solutions in {moqo_time:.1f}s")

                    # Save RandomMOQO results
                    for idx, ind in enumerate(moqo_pareto):
                        all_results.append({
                            'algorithm': 'RandomMOQO',
                            'k': k,
                            'level': level,
                            'query_type': qt,
                            'solution_id': idx,
                            'struct_id': ind.struct_id,
                            'assignment': str(ind.assignment),
                            'cost': ind.objectives[0] if ind.objectives else None,
                            'energy': ind.objectives[1] if ind.objectives else None,
                            'latency': ind.objectives[2] if ind.objectives else None,
                            'qoa': -ind.objectives[3] if ind.objectives else None,
                            'runtime': moqo_time
                        })

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = "experiment_results_k_levels.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n{'=' * 80}")
    print('EXPERIMENT COMPLETED')
    print('=' * 80)
    print(f'Total results: {len(results_df)}')
    print(f'Saved to: {output_file}')

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print('SUMMARY STATISTICS')
    print('=' * 80)

    for algo in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Total solutions: {len(algo_data)}")
        print(f"  Average solutions per config: {len(algo_data) / (len(k_values) * len(levels) * len(QUERY_TYPES)):.1f}")

        # Group by k and level
        grouped = algo_data.groupby(['k', 'level']).size()
        print(f"\n  Solutions by (k, level):")
        for (k_val, lv), count in grouped.items():
            print(f"    k={k_val}, level={lv}: {count} solutions")

    print(f"\n{'=' * 80}")

    return results_df


# =============================================================================
# Diversity Experiment: Evaluate effect of LLM diversity on QoA and resource usage
# =============================================================================

def calculate_diversity(assignment):
    """
    Calculate diversity level d for a plan.

    Args:
        assignment: list or tuple of LLM model indices

    Returns:
        int: count of unique models invoked (diversity level d)
    """
    return len(set(assignment))


def run_diversity_experiment(args, query_tokens, blending_prompt_tokens, ctx_tokens):
    """
    Run diversity experiment to evaluate effect of LLM diversity on QoA and resource usage.

    Settings (based on Section 6.2):
    - K = 5 (maximum number of operations per plan)
    - Vary diversity level d from 1 to min(K, |L|)
    - d = count of unique models invoked in a plan

    For each diversity level:
    - Generate/select plans with exactly that diversity level
    - Run both NSGA-II and RandomMOQO
    - Track QoA, cost, latency, energy
    - Analyze correlation between diversity and metrics
    """

    # Fixed K=5 for this experiment
    K = 5

    # Determine diversity levels to test
    if args.diversity_levels:
        diversity_levels = [int(d.strip()) for d in args.diversity_levels.split(',')]
    else:
        # Default: all diversity levels from 1 to min(K, |L|)
        # Assuming |L| = 6 models (0-5) based on typical LLM pool size
        num_models = 6  # TODO: could extract from df_history or make configurable
        max_diversity = min(K, num_models)
        diversity_levels = list(range(1, max_diversity + 1))

    # Use all PerfDB levels or specified levels
    if args.levels:
        levels = [int(lv.strip()) for lv in args.levels.split(',')]
    else:
        levels = [0, 1, 2, 3, 4]  # All levels by default

    print('=' * 80)
    print('DIVERSITY EXPERIMENT: Effect of LLM Diversity on QoA and Resource Usage')
    print('=' * 80)
    print(f'K (max operations): {K}')
    print(f'Query types: {QUERY_TYPES}')
    print(f'Diversity levels (d): {diversity_levels}')
    print(f'PerfDB levels: {levels}')
    print(f'Algorithm: {"Both" if args.algorithm == "both" else args.algorithm.upper()}')
    print('=' * 80)
    print()

    # Results storage
    all_results = []

    # Determine which algorithms to run
    run_nsga = args.algorithm in ["both", "nsga"]
    run_moqo = args.algorithm in ["both", "moqo"]

    # Main experiment loop
    config_num = 0
    total_configs = len(diversity_levels) * len(levels)

    for level in levels:
        # Load data for this PerfDB level
        level_file = f"{args.level_dir}/level_{level}_data.csv"
        print(f"\n{'=' * 80}")
        print(f"Loading PerfDB Level {level}: {level_file}")
        print(f"{'=' * 80}")

        try:
            df_history = pd.read_csv(level_file)
            df_history['llm_assignments'] = (
                df_history['llm_assignments']
                .astype(str)
                .str.replace(r'[\(\)\s]', '', regex=True)
                .str.rstrip(',')
            )
            print(f"✓ Loaded {len(df_history)} records")
        except FileNotFoundError:
            print(f"✗ File not found: {level_file}")
            continue

        for diversity_d in diversity_levels:
            config_num += 1
            print(f"\n{'-' * 80}")
            print(f"Configuration {config_num}/{total_configs}: Diversity={diversity_d}, Level={level}")
            print(f"{'-' * 80}")

            for qt in QUERY_TYPES:
                print(f"\n  Query: {qt}")

                # Clear evaluation cache
                _evaluation_cache.clear()

                nsga_time = None
                nsga_solutions = 0
                moqo_time = None
                moqo_solutions = 0

                # ===== Run NSGA-II =====
                if run_nsga:
                    print(f"    [NSGA-II] Starting (K={K}, diversity={diversity_d})...")
                    start = time.time()

                    # Run NSGA-II with K=5
                    nsga_pareto = nsga2_optimize(
                        query_tokens,
                        blending_prompt_tokens,
                        ctx_tokens,
                        df_history,
                        pop_size=POP_SIZE,
                        generations=GENERATIONS,
                        max_nodes=K,
                        query_type=qt
                    )

                    nsga_time = time.time() - start

                    # Filter solutions by diversity level
                    nsga_pareto_filtered = [
                        ind for ind in nsga_pareto
                        if calculate_diversity(ind.assignment) == diversity_d
                    ]

                    nsga_solutions = len(nsga_pareto_filtered)

                    print(f"    [NSGA-II] Completed: {len(nsga_pareto)} total solutions, "
                          f"{nsga_solutions} with diversity={diversity_d} in {nsga_time:.1f}s")

                    # Save NSGA-II results for this diversity level
                    for idx, ind in enumerate(nsga_pareto_filtered):
                        all_results.append({
                            'algorithm': 'NSGA-II',
                            'k': K,
                            'diversity': diversity_d,
                            'level': level,
                            'query_type': qt,
                            'solution_id': idx,
                            'struct_id': ind.struct_id,
                            'assignment': str(ind.assignment),
                            'cost': ind.objectives[0] if ind.objectives else None,
                            'energy': ind.objectives[1] if ind.objectives else None,
                            'latency': ind.objectives[2] if ind.objectives else None,
                            'qoa': -ind.objectives[3] if ind.objectives else None,
                            'runtime': nsga_time
                        })

                # ===== Run RandomMOQO =====
                if run_moqo:
                    _evaluation_cache.clear()

                    # Determine timeout
                    if args.moqo_time:
                        moqo_timeout = args.moqo_time
                    elif nsga_time:
                        moqo_timeout = int(nsga_time)
                    else:
                        moqo_timeout = 60

                    print(f"    [RandomMOQO] Starting (K={K}, diversity={diversity_d}, timeout: {moqo_timeout}s)...")
                    start = time.time()

                    moqo_pareto = RandomMOQO(
                        query_type=qt,
                        df_history=df_history,
                        timeout_seconds=moqo_timeout,
                        max_nodes=K,
                        use_adaptive_precision=True,
                        alpha_start=2.0,
                        alpha_end=1.0
                    )

                    moqo_time = time.time() - start

                    # Filter solutions by diversity level
                    moqo_pareto_filtered = [
                        ind for ind in moqo_pareto
                        if calculate_diversity(ind.assignment) == diversity_d
                    ]

                    moqo_solutions = len(moqo_pareto_filtered)

                    print(f"    [RandomMOQO] Completed: {len(moqo_pareto)} total solutions, "
                          f"{moqo_solutions} with diversity={diversity_d} in {moqo_time:.1f}s")

                    # Save RandomMOQO results for this diversity level
                    for idx, ind in enumerate(moqo_pareto_filtered):
                        all_results.append({
                            'algorithm': 'RandomMOQO',
                            'k': K,
                            'diversity': diversity_d,
                            'level': level,
                            'query_type': qt,
                            'solution_id': idx,
                            'struct_id': ind.struct_id,
                            'assignment': str(ind.assignment),
                            'cost': ind.objectives[0] if ind.objectives else None,
                            'energy': ind.objectives[1] if ind.objectives else None,
                            'latency': ind.objectives[2] if ind.objectives else None,
                            'qoa': -ind.objectives[3] if ind.objectives else None,
                            'runtime': moqo_time
                        })

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = "diversity_experiment_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n{'=' * 80}")
    print('DIVERSITY EXPERIMENT COMPLETED')
    print('=' * 80)
    print(f'Total results: {len(results_df)}')
    print(f'Saved to: {output_file}')

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print('SUMMARY STATISTICS')
    print('=' * 80)

    for algo in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Total solutions: {len(algo_data)}")

        # Group by diversity and level
        grouped = algo_data.groupby(['diversity', 'level']).size()
        print(f"\n  Solutions by (diversity, level):")
        for (d_val, lv), count in grouped.items():
            print(f"    diversity={d_val}, level={lv}: {count} solutions")

        # Statistics by diversity level
        print(f"\n  Metrics by diversity level:")
        for d_val in diversity_levels:
            d_data = algo_data[algo_data['diversity'] == d_val]
            if len(d_data) > 0:
                print(f"    diversity={d_val}:")
                print(f"      Solutions: {len(d_data)}")
                print(f"      Avg QoA: {d_data['qoa'].mean():.4f} (range: {d_data['qoa'].min():.4f}-{d_data['qoa'].max():.4f})")
                print(f"      Avg Cost: ${d_data['cost'].mean():.6f} (range: ${d_data['cost'].min():.6f}-${d_data['cost'].max():.6f})")
                print(f"      Avg Latency: {d_data['latency'].mean():.2f}s (range: {d_data['latency'].min():.2f}-{d_data['latency'].max():.2f}s)")
                print(f"      Avg Energy: {d_data['energy'].mean():.2f} (range: {d_data['energy'].min():.2f}-{d_data['energy'].max():.2f})")

    print(f"\n{'=' * 80}")

    # Correlation analysis
    print(f"\n{'=' * 80}")
    print('CORRELATION ANALYSIS: Diversity vs Metrics')
    print('=' * 80)

    if len(results_df) > 0:
        corr_qoa = results_df[['diversity', 'qoa']].corr().iloc[0, 1]
        corr_cost = results_df[['diversity', 'cost']].corr().iloc[0, 1]
        corr_latency = results_df[['diversity', 'latency']].corr().iloc[0, 1]
        corr_energy = results_df[['diversity', 'energy']].corr().iloc[0, 1]

        print(f"\nPearson Correlation Coefficients:")
        print(f"  Diversity vs QoA:     {corr_qoa:+.4f}")
        print(f"  Diversity vs Cost:    {corr_cost:+.4f}")
        print(f"  Diversity vs Latency: {corr_latency:+.4f}")
        print(f"  Diversity vs Energy:  {corr_energy:+.4f}")

        print(f"\nInterpretation:")
        if abs(corr_qoa) > 0.5:
            print(f"  - {'Strong positive' if corr_qoa > 0 else 'Strong negative'} correlation between diversity and QoA")
        elif abs(corr_qoa) > 0.3:
            print(f"  - {'Moderate positive' if corr_qoa > 0 else 'Moderate negative'} correlation between diversity and QoA")
        else:
            print(f"  - Weak correlation between diversity and QoA")

    print(f"\n{'=' * 80}")

    return results_df


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Run optimization (example with smaller population/generations for demonstration)
    query_tokens = 215
    blending_prompt_tokens = 26
    ctx_tokens = 39
    load_from_pickle = False

    # Parse args first so we can use --levels
    args = parse_args()

    # Load dataset from Levels folder based on --levels argument
    # For normal mode, use the first level from the comma-separated list
    if hasattr(args, 'levels') and args.levels:
        level = int(args.levels.split(',')[0].strip())
    else:
        level = 4
    FILEPATH_TO_HISTORY_FILE = f"Levels/level_{level}_data.csv"

    if not os.path.exists(FILEPATH_TO_HISTORY_FILE):
        print(f"Error: Dataset not found: {FILEPATH_TO_HISTORY_FILE}")
        print(f"Available levels: 0, 1, 2, 3, 4")
        print(f"Files should be in: Levels/")
        sys.exit(1)

    print(f"Loading dataset: {FILEPATH_TO_HISTORY_FILE}")
    df_history = pd.read_csv(FILEPATH_TO_HISTORY_FILE)
    print(f"✓ Loaded {len(df_history)} records\n")

    df_history['llm_assignments'] = (
        df_history['llm_assignments']
        .astype(str)
        .str.replace(r'[\(\)\s]', '', regex=True)  # remove ( ) and spaces
        .str.rstrip(',')  # remove trailing comma if any
    )
    if args.config:
        load_config(args.config)

    # Override MAX_NODES from command line if provided
    if args.max_nodes is not None:
        MAX_NODES = args.max_nodes
        print(f'Using max_nodes={MAX_NODES} from command line')

    # =============================================================================
    #                  Experiment Mode vs Normal Execution
    # =============================================================================

    if args.diversity_experiment:
        # Run diversity experiment (K=5, vary diversity level d)
        run_diversity_experiment(args, query_tokens, blending_prompt_tokens, ctx_tokens)
        exit(0)

    if args.experiment:
        # Run full experiment varying k and PerfDB levels
        run_experiment(args, query_tokens, blending_prompt_tokens, ctx_tokens)
        exit(0)

    # =============================================================================
    #                  NSGA-II and/or RandomMOQO Execution (Normal Mode)
    # =============================================================================

    # Determine which algorithms to run
    run_nsga = args.algorithm in ["both", "nsga", "all"]
    run_moqo = args.algorithm in ["both", "moqo", "all"]
    run_fptas = args.algorithm in ["fptas", "all"]

    # Print header
    print('=' * 80)
    if args.algorithm == "all":
        print('COMPARISON: NSGA-II vs RandomMOQO vs FPTAS')
    elif args.algorithm == "both":
        print('COMPARISON: NSGA-II vs RandomMOQO')
    elif args.algorithm == "nsga":
        print('RUNNING: NSGA-II ONLY')
    elif args.algorithm == "moqo":
        print('RUNNING: RandomMOQO ONLY')
    elif args.algorithm == "fptas":
        print('RUNNING: FPTAS ONLY')
    print('=' * 80)
    print(f'Query types: {QUERY_TYPES}')
    print(f'Repetitions: {REPETITIONS}')
    if not run_nsga and args.moqo_time:
        print(f'RandomMOQO timeout: {args.moqo_time}s (fixed)')
    print()

    # Results storage
    nsga_results = []
    moqo_results = []
    fptas_results = []
    nsga_times = []  # Track NSGA-II times for averaging

    for qt in QUERY_TYPES:
        for run_idx in range(1, REPETITIONS + 1):
            print('\n' + '=' * 80)
            print(f'Query Type: {qt}, Iteration {run_idx}/{REPETITIONS}')
            print('=' * 80)

            nsga_time = None
            nsga_pareto = None
            nsga_unique_objectives = set()

            # ===== Run NSGA-II =====
            if run_nsga:
                step_label = '[1/2]' if run_moqo else '[NSGA-II]'
                print(f'\n{step_label} Running NSGA-II...')
                print('-' * 80)
                start = time.time()

                nsga_pareto = nsga2_optimize(
                    query_tokens,
                    blending_prompt_tokens,
                    ctx_tokens,
                    df_history,
                    pop_size=POP_SIZE,
                    generations=GENERATIONS,
                    max_nodes=MAX_NODES,
                    query_type=qt
                )

                nsga_time = time.time() - start
                nsga_times.append(nsga_time)  # Track for averaging

                print(f'\nNSGA-II Results:')
                print(f'  Time: {nsga_time:.1f}s')
                print(f'  Pareto solutions: {len(nsga_pareto)}')

                # Count unique solutions by objectives
                for ind in nsga_pareto:
                    c, t, e, q = ind.metrics
                    nsga_unique_objectives.add((round(c, 8), round(t, 4), round(e, 4), round(q, 6)))
                print(f'  Unique solutions (by objectives): {len(nsga_unique_objectives)}')

                # Store results
                for ind in nsga_pareto:
                    c, t, e, q = ind.metrics
                    nsga_results.append({
                        'algorithm': 'NSGA-II',
                        'query_type': qt,
                        'iteration': run_idx,
                        'struct_id': ind.struct_id,
                        'assignment': ind.assignment,
                        'cost': c,
                        'latency': t,
                        'energy': e,
                        'qoa': q,
                        'run_time_seconds': nsga_time
                    })

            # ===== Run RandomMOQO =====
            if run_moqo:
                step_label = '[2/2]' if run_nsga else '[RandomMOQO]'
                print(f'\n{step_label} Running RandomMOQO...')
                print('-' * 80)

                # Clear evaluation cache between runs
                _evaluation_cache.clear()

                # Determine timeout for RandomMOQO
                if args.moqo_time:
                    # Use fixed time from command line
                    moqo_timeout = args.moqo_time
                elif nsga_time:
                    # Use time from current NSGA-II run
                    moqo_timeout = int(nsga_time)
                elif nsga_times:
                    # Use average from previous NSGA-II runs
                    moqo_timeout = int(sum(nsga_times) / len(nsga_times))
                else:
                    # Fallback default
                    moqo_timeout = 60

                start = time.time()

                moqo_pareto = RandomMOQO(
                    query_type=qt,
                    df_history=df_history,
                    timeout_seconds=moqo_timeout,
                    max_nodes=MAX_NODES
                )

                moqo_time = time.time() - start
                print(f'\nRandomMOQO Results:')
                print(f'  Time: {moqo_time:.1f}s')
                print(f'  Pareto solutions: {len(moqo_pareto)}')

                # Count unique solutions
                moqo_unique_objectives = set()
                for ind in moqo_pareto:
                    c, t, e, q = ind.metrics
                    moqo_unique_objectives.add((round(c, 8), round(t, 4), round(e, 4), round(q, 6)))
                print(f'  Unique solutions (by objectives): {len(moqo_unique_objectives)}')

                # Store results
                for ind in moqo_pareto:
                    c, t, e, q = ind.metrics
                    moqo_results.append({
                        'algorithm': 'RandomMOQO',
                        'query_type': qt,
                        'iteration': run_idx,
                        'struct_id': ind.struct_id,
                        'assignment': ind.assignment,
                        'cost': c,
                        'latency': t,
                        'energy': e,
                        'qoa': q,
                        'run_time_seconds': moqo_time
                    })

            # ===== Run FPTAS =====
            if run_fptas:
                step_label = '[FPTAS]'
                if run_nsga and run_moqo:
                    step_label = '[3/3]'
                elif run_nsga or run_moqo:
                    step_label = '[2/2]'

                print(f'\n{step_label} Running FPTAS...')
                print('-' * 80)

                # Clear evaluation cache between runs
                _evaluation_cache.clear()

                # Import FPTAS
                from fptas import FPTAS

                start = time.time()

                fptas_pareto = FPTAS(
                    query_type=qt,
                    df_history=df_history,
                    max_nodes=MAX_NODES,
                    epsilon=args.epsilon,
                    verbose=True
                )

                fptas_time = time.time() - start
                print(f'\nFPTAS Results:')
                print(f'  Time: {fptas_time:.1f}s')
                print(f'  ε-Pareto solutions: {len(fptas_pareto)}')
                print(f'  ε: {args.epsilon} ({args.epsilon*100:.0f}% approximation)')

                # Count unique solutions
                fptas_unique_objectives = set()
                for ind in fptas_pareto:
                    c, t, e, q = ind.metrics
                    fptas_unique_objectives.add((round(c, 8), round(t, 4), round(e, 4), round(q, 6)))
                print(f'  Unique solutions (by objectives): {len(fptas_unique_objectives)}')

                # Store results
                for ind in fptas_pareto:
                    c, t, e, q = ind.metrics
                    fptas_results.append({
                        'algorithm': 'FPTAS',
                        'query_type': qt,
                        'iteration': run_idx,
                        'struct_id': ind.struct_id,
                        'assignment': ind.assignment,
                        'cost': c,
                        'latency': t,
                        'energy': e,
                        'qoa': q,
                        'epsilon': args.epsilon,
                        'run_time_seconds': fptas_time
                    })

            # ===== Comparison =====
            if run_nsga and run_moqo:
                print('\n' + '=' * 80)
                print('COMPARISON SUMMARY')
                print('=' * 80)
                print(f'{"Metric":<30} {"NSGA-II":<20} {"RandomMOQO":<20}')
                print('-' * 80)
                print(f'{"Runtime (seconds)":<30} {nsga_time:<20.1f} {moqo_time:<20.1f}')
                print(f'{"Total solutions":<30} {len(nsga_pareto):<20} {len(moqo_pareto):<20}')
                print(f'{"Unique solutions":<30} {len(nsga_unique_objectives):<20} {len(moqo_unique_objectives):<20}')
                print(f'{"Duplication rate":<30} {(1-len(nsga_unique_objectives)/len(nsga_pareto))*100:<20.1f}% {(1-len(moqo_unique_objectives)/max(1, len(moqo_pareto)))*100:<20.1f}%')
                print('=' * 80)

    # Print average NSGA-II time if multiple runs
    if nsga_times and len(nsga_times) > 1:
        avg_nsga_time = sum(nsga_times) / len(nsga_times)
        print('\n' + '=' * 80)
        print('NSGA-II TIMING SUMMARY')
        print('=' * 80)
        print(f'Total runs: {len(nsga_times)}')
        print(f'Times: {[f"{t:.1f}s" for t in nsga_times]}')
        print(f'Average time: {avg_nsga_time:.1f}s')
        print(f'Min time: {min(nsga_times):.1f}s')
        print(f'Max time: {max(nsga_times):.1f}s')
        print('=' * 80)
        print(f'\nℹ️  Use --algorithm moqo --moqo-time {avg_nsga_time:.0f} to run RandomMOQO only with this timeout')

    # Save results to separate files
    print('\n' + '=' * 80)
    print('SAVING RESULTS')
    print('=' * 80)

    # Save NSGA-II results
    if nsga_results:
        df_nsga = pd.DataFrame(nsga_results)
        nsga_file = RESULTS_FILE.replace('.csv', '_nsga2.csv')
        df_nsga.to_csv(nsga_file, index=False)
        print(f"NSGA-II: {len(df_nsga)} results saved to {nsga_file}")

    # Save RandomMOQO results
    if moqo_results:
        df_moqo = pd.DataFrame(moqo_results)
        moqo_file = RESULTS_FILE.replace('.csv', '_randommoqo.csv')
        df_moqo.to_csv(moqo_file, index=False)
        print(f"RandomMOQO: {len(df_moqo)} results saved to {moqo_file}")

    # Save FPTAS results
    if fptas_results:
        df_fptas = pd.DataFrame(fptas_results)
        fptas_file = RESULTS_FILE.replace('.csv', '_fptas.csv')
        df_fptas.to_csv(fptas_file, index=False)
        print(f"FPTAS: {len(df_fptas)} results saved to {fptas_file}")

    # Save combined results
    all_results = nsga_results + moqo_results + fptas_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(RESULTS_FILE, index=False)
        print(f"Combined: {len(df_all)} results saved to {RESULTS_FILE}")

    print('=' * 80)
    # python nsga_final.py --config config.json


