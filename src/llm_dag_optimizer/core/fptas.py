#!/usr/bin/env python3
"""
PROPER True DP FPTAS — Enumerate ALL DAG plan structures bottom-up (k=1..K),
then inherit assignments from substructures (paper-style), with ε-pruning.

Key approach:
- Build DP[k] from DP[k-1] by adding ONE new node (k-1) and choosing ANY
  subset of predecessors S ⊆ {0..k-2} that feed into it (pred_mask loop).
- Blending is implicit: any node with in-degree > 1 must be model 5.
- Canonicalization is applied at every child to dedup isomorphic DAGs.
- IMPORTANT: because canonicalization relabels nodes, we remap assignments using the
  canonical permutation (so inheritance is correct).

DP structure:
  DP[k][connectivity_map] = (canonical_graph, [Assignment ...])

This enumerates ALL non-isomorphic DAG structures while using TRUE DP for assignments
(inherit from subproblems, don't re-enumerate 5^k combinations).
"""

import time
import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import networkx as nx
import pandas as pd

from ..structures.individual import Individual
from ..structures.dag import canonical_representation


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class Assignment:
    """Assignment of LLMs to nodes with evaluated metrics."""
    node_llms: Tuple[int, ...]                    # model per node (in CANONICAL node order)
    metrics: Tuple[float, float, float, float]    # (cost, latency, energy, qoa)

    def __hash__(self):
        return hash(self.node_llms)


# =============================================================================
# Canonicalization helpers (need permutation to remap assignments)
# =============================================================================

def _adj_from_graph(G: nx.DiGraph, k: int) -> List[List[int]]:
    """Convert graph to adjacency matrix (assumes nodes 0..k-1)."""
    adj = [[0] * k for _ in range(k)]
    for u, v in G.edges():
        adj[u][v] = 1
    return adj


@lru_cache(maxsize=200000)
def _canonical_mask_and_perm_cached(adj_tuple: Tuple[Tuple[int, ...], ...], k: int) -> Tuple[int, Tuple[int, ...]]:
    """
    Returns (best_mask, best_perm) where:
      best_perm[new_pos] = old_node_id  (a valid topo order that maximizes lexicographic mask)

    This is the paper's canonical representation that ensures isomorphic DAGs have the same key.
    """
    adj = [list(row) for row in adj_tuple]
    nodes = list(range(k))
    best_mask = -1
    best_perm: Optional[Tuple[int, ...]] = None

    for perm in itertools.permutations(nodes):
        pos = {node: idx for idx, node in enumerate(perm)}

        # Valid topological order check
        valid = True
        for i in range(k):
            for j in range(k):
                if adj[i][j] == 1 and pos[i] > pos[j]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Compute mask in this perm-order
        new_mask = 0
        bit = 0
        for a in range(k - 1):
            for b in range(a + 1, k):
                x = perm[a]
                y = perm[b]
                if adj[x][y] == 1:
                    new_mask |= (1 << bit)
                bit += 1

        if new_mask > best_mask:
            best_mask = new_mask
            best_perm = perm

    assert best_perm is not None  # k>=1 always has at least one topo order
    return best_mask, best_perm


def canonical_mask_and_perm_from_graph(G: nx.DiGraph) -> Tuple[int, Tuple[int, ...]]:
    """
    Get canonical connectivity_map + permutation.
    Permutation is needed to remap assignments correctly after canonicalization.
    """
    k = G.number_of_nodes()
    # Ensure labeled 0..k-1
    if list(G.nodes()) != list(range(k)):
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping, copy=True)

    adj = _adj_from_graph(G, k)
    adj_tuple = tuple(tuple(int(v) for v in row) for row in adj)
    return _canonical_mask_and_perm_cached(adj_tuple, k)


def mask_to_graph(k: int, mask: int) -> nx.DiGraph:
    """
    Build the canonical-labeled graph from its upper-triangular bitmask.
    Bit order: (0,1),(0,2)...(0,k-1),(1,2)...(k-2,k-1)
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(k))
    bit = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if mask & (1 << bit):
                G.add_edge(i, j)
            bit += 1
    return G


# =============================================================================
# Evaluation and Validation
# =============================================================================

def evaluate_plan(
    G: nx.DiGraph,
    assignment: List[int],
    query_type: str,
    df_history: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """
    Evaluate a plan (graph + assignment) using historical data.

    Uses the same 3-tier evaluation as NSGA-II and MOQO:
    1. Exact PerfDB lookup
    2. Blend operation lookup
    3. Fallback estimator (no fuzzy matching)

    Returns:
        (cost, latency, energy, qoa)
    """
    from old.main import evaluate_individual_V2

    # Convert graph to struct_id for evaluate_individual_V2
    k = G.number_of_nodes()
    struct_id = 0
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if G.has_edge(i, j):
                struct_id |= (1 << bit_index)
            bit_index += 1

    llm_assignment = [str(a) for a in assignment]

    try:
        return evaluate_individual_V2(
            struct_id, llm_assignment, query_type,
            query_tokens=215, blending_prompt_tokens=26,
            ctx_tokens=39, df_history=df_history
        )
    except Exception:
        return (float("inf"), float("inf"), float("inf"), 0.0)


def validate_assignment(G: nx.DiGraph, assignment: List[int]) -> bool:
    """
    Validate blend constraint:
      indeg(node) > 1  => assignment[node] must be 5
      indeg(node) <= 1 => assignment[node] must be 0..4 (not 5)
    """
    for node in G.nodes():
        indeg = G.in_degree(node)
        llm = assignment[node]
        if indeg > 1:
            if llm != 5:
                return False
        else:
            if llm == 5:
                return False
    return True


from typing import Tuple

def epsilon_dominates_strict(m1: Tuple[float,float,float,float],
                            m2: Tuple[float,float,float,float],
                            eps: float,
                            tol: float = 1e-12) -> bool:
    """
    m1 ε-dominates m2 if:
      - m1 is within (1+ε) factor on all minimized metrics (cost, latency, energy)
      - m1 is within 1/(1+ε) on QoA (maximize)
      - and m1 is strictly better than m2 in at least one objective (without ε relaxation)

    This prevents mutual domination: two solutions cannot both dominate each other.
    """
    c1,l1,e1,q1 = m1
    c2,l2,e2,q2 = m2
    a = 1.0 + eps

    # no-worse within ε (multiplicative)
    no_worse = (
        c1 <= a * c2 + tol and
        l1 <= a * l2 + tol and
        e1 <= a * e2 + tol and
        q1 >= (q2 / a) - tol
    )
    if not no_worse:
        return False

    # strictly better in at least one dimension (compare to original values, not ε bounds)
    strictly_better = (
        c1 < c2 - tol or
        l1 < l2 - tol or
        e1 < e2 - tol or
        q1 > q2 + tol
    )
    return strictly_better

def _clamp_metrics(m, floor=1e-12):
    c,l,e,q = m
    return (max(c,floor), max(l,floor), max(e,floor), q)


def epsilon_prune_assignments(assignments, eps, disable_pruning=False):
    """
    Paper-correct ε-nondominated set computation via single pairwise scan.

    Algorithm: Mark all points ε-dominated by any other point, remove marked.
    Guarantees: every removed point is ε-dominated by some point in final kept set.

    Corrected ε-dominance: m1 ε-dominates m2 if:
    - m1 is within (1+ε) factor on all objectives (no worse)
    - m1 is strictly better than ε-relaxed bound in at least one objective

    Complexity: O(n²) single pass per bucket.
    """
    if disable_pruning:
        return assignments
    if not assignments:
        return []

    a = 1.0 + eps
    tol = 1e-12

    def eps_dom_strict(m1, m2):
        """
        m1 ε-dominates m2 using corrected criterion.

        No-worse: m1 within (1+ε) factor on all objectives
        Strictly better: m1 strictly better than ε-relaxed bound on at least one objective
        """
        c1, l1, e1, q1 = m1
        c2, l2, e2, q2 = m2

        no_worse = (
            c1 <= a * c2 + tol and
            l1 <= a * l2 + tol and
            e1 <= a * e2 + tol and
            q1 >= q2 / a - tol
        )
        strictly = (
            c1 < a * c2 - tol or
            l1 < a * l2 - tol or
            e1 < a * e2 - tol or
            q1 > q2 / a + tol
        )
        return no_worse and strictly

    # Pre-clamp all metrics once (avoid redundant clamps in inner loop)
    clamped = [_clamp_metrics(a.metrics) for a in assignments]

    # Single pairwise scan: mark dominated points
    n = len(assignments)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue  # Already marked, skip

        for j in range(n):
            if i == j or dominated[j]:
                continue

            # Check if j ε-dominates i
            if eps_dom_strict(clamped[j], clamped[i]):
                dominated[i] = True
                break  # Found dominator, no need to check others

    # Return non-dominated assignments
    return [assignments[i] for i in range(n) if not dominated[i]]


def delta_qoa_prune_assignments(assignments, best_qoa_so_far, delta, disable_pruning=False):
    """
    Delta-QoA pruning (DEPRECATED - use parent-specific version instead).

    This global threshold version is too aggressive. Use delta_qoa_prune_parent_specific.
    """
    if disable_pruning:
        return assignments
    if not assignments:
        return []

    threshold = best_qoa_so_far + delta
    kept = [a for a in assignments if a.metrics[3] >= threshold]
    return kept


def delta_qoa_prune_parent_specific(parent_assign, new_llms_and_metrics, delta, disable_pruning=False):
    """
    Delta-QoA pruning (CORRECT): keep only children that improve over THEIR parent.

    Args:
        parent_assign: Parent Assignment object
        new_llms_and_metrics: List of (new_llm, canon_assign, metrics) tuples for children
        delta: Minimum QoA improvement required over parent
        disable_pruning: If True, return all

    Returns:
        List of (new_llm, canon_assign, metrics) that pass delta threshold
    """
    if disable_pruning:
        return new_llms_and_metrics
    if not new_llms_and_metrics:
        return []

    parent_qoa = parent_assign.metrics[3]
    threshold = parent_qoa + delta

    # Keep children that improve over this specific parent
    kept = [(llm, assign, metrics) for llm, assign, metrics in new_llms_and_metrics
            if metrics[3] >= threshold]

    return kept


# =============================================================================
# Admissible plan filter (use ONLY for extraction, not DP building)
# =============================================================================

def is_single_sink_all_reach_sink(G: nx.DiGraph) -> bool:
    """
    Check NSGA-style admissible structure:
      - only last node (k-1) is sink (outdeg=0)
      - all other nodes have outdeg>0
      - every node can reach the sink
    """
    k = G.number_of_nodes()

    # Get actual node list (in case graph isn't numbered 0..k-1)
    nodes = sorted(G.nodes())
    if len(nodes) != k:
        return False

    # Expect nodes to be 0..k-1 in canonical form
    if nodes != list(range(k)):
        return False

    sink = k - 1

    if G.out_degree(sink) != 0:
        return False

    for v in range(k - 1):
        if G.out_degree(v) == 0:
            return False

    for v in range(k - 1):
        if not nx.has_path(G, v, sink):
            return False

    return True


# =============================================================================
# DP Operation: extend by adding node with ANY predecessor subset
# =============================================================================

def extend_with_pred_mask(parent_graph: nx.DiGraph, k: int, pred_mask: int) -> nx.DiGraph:
    """
    Create child graph from parent by adding new node (k-1).
    Add edges i->(k-1) for each bit i set in pred_mask.

    Args:
        parent_graph: Graph with nodes 0..k-2
        k: Size of child graph
        pred_mask: Bitmask indicating which nodes feed into new node

    Returns:
        Child graph with k nodes
    """
    parent_k = parent_graph.number_of_nodes()
    if parent_k != k - 1:
        # Skip if parent doesn't match expected size
        # (can happen with mixed k values in DP table)
        return None

    child = parent_graph.copy()
    new_node = k - 1
    child.add_node(new_node)

    for i in range(k - 1):
        if (pred_mask >> i) & 1:
            child.add_edge(i, new_node)

    return child


# =============================================================================
# Main DP Algorithm
# =============================================================================

def FPTAS(
    query_type: str,
    df_history: pd.DataFrame,
    max_nodes: int = 6,
    epsilon: float = 0.05,
    delta: float = 0.05,
    verbose: bool = True,
    allow_empty_preds: bool = True,
    extract_only_admissible: bool = True,
    disable_pruning: bool = False,
    pruning_strategy: str = 'epsilon',  # 'epsilon', 'delta_qoa', or None
) -> List[Individual]:
    """
    Bottom-up DP FPTAS that enumerates ALL DAG structures while using TRUE DP for assignments.

    Process:
      DP[1] -> DP[2] -> ... -> DP[max_nodes]

    Each step:
      - Take each parent structure from DP[k-1]
      - Add node (k-1) with all possible predecessor subsets (pred_mask)
      - Canonicalize child structure
      - Inherit assignments from parent, add 5 choices for new node
      - Remap assignments to canonical node order
      - Apply pruning (ε-dominance or Delta-QoA)

    This gives:
      - Complete structure coverage (all non-isomorphic DAGs)
      - True DP for assignments (inherit, don't re-enumerate)
      - Correct canonicalization (assignments match canonical node order)

    Args:
        query_type: Query category for evaluation
        df_history: Historical metrics data
        max_nodes: Maximum nodes in final graph
        epsilon: Approximation parameter for ε-dominance (0.01 = 1%)
        delta: Minimum QoA improvement for Delta-QoA pruning
        verbose: Print progress
        allow_empty_preds: Allow new node with in-degree 0 during enumeration
        extract_only_admissible: Filter final results to single-sink structures
        disable_pruning: If True, disable all pruning
        pruning_strategy: 'epsilon' for ε-dominance, 'delta_qoa' for Delta-QoA, None for no pruning

    Returns:
        List of Individual objects representing final solutions
    """
    t0 = time.time()
    DP: Dict[int, Dict[int, Tuple[nx.DiGraph, List[Assignment]]]] = {}

    if verbose:
        print("\n" + "=" * 80)
        print("TRUE DP FPTAS — ALL DAG STRUCTURES (BOTTOM-UP)")
        print("=" * 80)
        print(f"Query type: {query_type}")
        print(f"Max nodes: {max_nodes}")
        if pruning_strategy == 'epsilon':
            print(f"Pruning: ε-dominance (ε={epsilon:.4f}, α={1+epsilon:.4f})")
        elif pruning_strategy == 'delta_qoa':
            print(f"Pruning: Delta-QoA (Δ={delta:.4f})")
        else:
            print(f"Pruning: Disabled")
        print(f"Allow empty predecessors: {allow_empty_preds}")
        print(f"Extract only admissible: {extract_only_admissible}")
        print()

    # ---------------------------
    # k=1 base case
    # ---------------------------
    G1 = nx.DiGraph()
    G1.add_node(0)

    key1, _ = canonical_mask_and_perm_from_graph(G1)
    G1_canon = mask_to_graph(1, key1)

    assigns1: List[Assignment] = []
    for llm in range(5):  # LLMs 0-4 allowed for single node (not blend)
        metrics = evaluate_plan(G1_canon, [llm], query_type, df_history)
        assigns1.append(Assignment((llm,), metrics))

    # Apply epsilon pruning (Delta-QoA: keep all k=1 solutions - no parents to compare)
    if pruning_strategy == 'epsilon':
        assigns1 = epsilon_prune_assignments(assigns1, epsilon, disable_pruning)

    DP[1] = {key1: (G1_canon, assigns1)}

    if verbose:
        print(f"k=1: structures={len(DP[1])}, assignments={sum(len(v[1]) for v in DP[1].values())}")

    # ---------------------------
    # k=2..max_nodes
    # ---------------------------
    for k in range(2, max_nodes + 1):
        DP[k] = {}
        start_mask = 0 if allow_empty_preds else 1

        # For each canonical parent structure
        for parent_key, (parent_graph, parent_assigns) in DP[k - 1].items():
            # Skip if parent doesn't match expected size
            if parent_graph.number_of_nodes() != k - 1:
                continue

            # Try all predecessor subsets into new node (k-1)
            for pred_mask in range(start_mask, 1 << (k - 1)):
                child_pre = extend_with_pred_mask(parent_graph, k, pred_mask)

                if child_pre is None:
                    continue

                # Canonicalize + get permutation to remap assignments
                child_key, perm = canonical_mask_and_perm_from_graph(child_pre)

                # Canonical rep graph for this key (stable)
                child_graph = mask_to_graph(k, child_key)

                # Determine allowed LLMs for new node based on in-degree
                indeg_new = child_pre.in_degree(k - 1)
                allowed_new_llms = [5] if indeg_new > 1 else list(range(5))

                # Build inherited assignments
                new_assigns: List[Assignment] = []
                for pa in parent_assigns:
                    # For Delta-QoA: prune immediately during evaluation
                    if pruning_strategy == 'delta_qoa':
                        threshold = pa.metrics[3] + delta if not disable_pruning else float('-inf')
                        for new_llm in allowed_new_llms:
                            old_assign = list(pa.node_llms) + [new_llm]
                            canon_assign = tuple(old_assign[perm[new_pos]] for new_pos in range(k))

                            if not validate_assignment(child_graph, list(canon_assign)):
                                continue

                            metrics = evaluate_plan(child_graph, list(canon_assign), query_type, df_history)

                            # Prune immediately: skip if doesn't improve by delta
                            if metrics[3] < threshold:
                                continue

                            new_assigns.append(Assignment(canon_assign, metrics))
                    else:
                        # For epsilon or no pruning: generate all children
                        for new_llm in allowed_new_llms:
                            old_assign = list(pa.node_llms) + [new_llm]
                            canon_assign = tuple(old_assign[perm[new_pos]] for new_pos in range(k))

                            if not validate_assignment(child_graph, list(canon_assign)):
                                continue

                            metrics = evaluate_plan(child_graph, list(canon_assign), query_type, df_history)
                            new_assigns.append(Assignment(canon_assign, metrics))

                if not new_assigns:
                    continue

                # Apply epsilon pruning (Delta-QoA already done per-parent above)
                if pruning_strategy == 'epsilon':
                    new_assigns = epsilon_prune_assignments(new_assigns, epsilon, disable_pruning)

                # Merge into DP[k][child_key]
                if child_key not in DP[k]:
                    DP[k][child_key] = (child_graph, new_assigns)
                else:
                    _, existing = DP[k][child_key]
                    existing.extend(new_assigns)
                    # Apply epsilon pruning after merge (Delta-QoA doesn't need merge pruning)
                    if pruning_strategy == 'epsilon':
                        existing = epsilon_prune_assignments(existing, epsilon, disable_pruning)
                    DP[k][child_key] = (child_graph, existing)

        if verbose:
            total_structs = len(DP[k])
            total_assigns = sum(len(v[1]) for v in DP[k].values())

            # Count admissible structures for reporting (doesn't affect DP)
            admissible_structs = sum(1 for key, (g, _) in DP[k].items()
                                    if is_single_sink_all_reach_sink(g))

            print(f"k={k}: structures={total_structs}, assignments={total_assigns}, "
                  f"admissible={admissible_structs}")

    # ---------------------------
    # Extract Individuals from ALL k layers (1 to max_nodes)
    # ---------------------------
    solutions: List[Individual] = []

    # Extract from all k values to get complete Pareto frontier
    for k in range(1, max_nodes + 1):
        for struct_key, (graph, assigns) in DP[k].items():
            if extract_only_admissible and not is_single_sink_all_reach_sink(graph):
                continue

            for a in assigns:
                ind = Individual(struct_key, list(a.node_llms))
                ind.cost, ind.latency, ind.energy, ind.qoa = a.metrics
                ind.metrics = a.metrics
                ind.num_nodes = k  # Track number of nodes
                solutions.append(ind)

    if verbose:
        dt = time.time() - t0
        print("-" * 80)
        print(f"Extracted solutions: {len(solutions)}")
        # Show breakdown by k
        k_counts = {}
        for s in solutions:
            k_counts[s.num_nodes] = k_counts.get(s.num_nodes, 0) + 1
        print(f"Solutions by k: {dict(sorted(k_counts.items()))}")
        print(f"Unique structures at k={max_nodes}: {len(DP[max_nodes])}")
        print(f"Time: {dt:.2f}s")
        print("=" * 80 + "\n")

    return solutions


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing True DP FPTAS...")
    df_history = pd.read_csv("Levels/level_4_data.csv")

    sols = FPTAS(
        query_type="Art",
        df_history=df_history,
        max_nodes=6,
        epsilon=0.05,
        verbose=True,
        allow_empty_preds=True,
        extract_only_admissible=True,
    )

    if sols:
        print(f"\nFinal results:")
        print(f"  Solutions: {len(sols)}")
        print(f"  QoA range: [{min(s.qoa for s in sols):.4f}, {max(s.qoa for s in sols):.4f}]")
    else:
        print("No solutions extracted (check filters / evaluation).")

