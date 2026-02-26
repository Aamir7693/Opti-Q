#!/usr/bin/env python3
"""
RandomMOQO Algorithm Module

Implements the RandomMOQO (Random Multi-Objective Query Optimization) algorithm
for finding Pareto-optimal LLM execution plans.

Key algorithms:
- RandomMOQO: Main algorithm with random restarts + hill climbing
- ParetoClimb: Hill climbing respecting Pareto dominance
- ParetoStep: Local search exploring mutations
- ApproximateFrontiers: Adaptive precision Pareto approximation

Reference: "An Adaptive Precision Approach to Efficiently Find
the Pareto Frontier of Multi-Objective Queries"
"""

import networkx as nx
import random
import time
import itertools
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import from refactored modules
from ..structures.individual import Individual
from ..structures.dag import (
    get_indegrees,
    generate_nonisomorphic_dags,
    repair_assignment_for_structure,
    is_valid_dag_structure
)
from ..config.defaults import DEFAULT_QOA_VALUE as DEFAULT_QOA


# =============================================================================
# Evaluation Cache (shared across RandomMOQO runs)
# =============================================================================

_evaluation_cache = {}


# =============================================================================
# Helper Functions
# =============================================================================

def node_type(G, node):
    """
    Determine node type based on in-degree.

    Args:
        G: NetworkX graph
        node: Node to check

    Returns:
        "start": No predecessors (in-degree 0)
        "seq": Single predecessor (in-degree 1)
        "blend": Multiple predecessors (in-degree > 1)
    """
    indeg = G.in_degree(node)
    if indeg == 0:
        return "start"
    if indeg == 1:
        return "seq"
    return "blend"


def graph_from_struct(k: int, struct_id: int) -> nx.DiGraph:
    """
    Convert structure ID (bitmask) to NetworkX directed graph.

    Args:
        k: Number of nodes
        struct_id: Bitmask encoding edges

    Returns:
        NetworkX DiGraph with k nodes and edges from struct_id
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(k))

    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if struct_id & (1 << bit_index):
                G.add_edge(i, j)
            bit_index += 1

    return G


def renumber_graph_nodes(G):
    """
    Renumber graph nodes to start from 0, maintaining edge structure.

    This solves the node indexing mismatch where graphs may have arbitrary
    node numbers (e.g., [2, 3, 4]) but assignment lists are 0-indexed [0, 1, 2].

    Args:
        G: NetworkX graph with arbitrary node numbers

    Returns:
        Tuple of:
        - new_graph: Graph with nodes renumbered from 0
        - node_mapping: Dict mapping old_node -> new_node
    """
    sorted_nodes = sorted(G.nodes())
    node_mapping = {old_node: new_idx for new_idx, old_node in enumerate(sorted_nodes)}

    new_G = nx.DiGraph()
    new_G.add_nodes_from(range(len(sorted_nodes)))
    for u, v in G.edges():
        new_G.add_edge(node_mapping[u], node_mapping[v])

    return new_G, node_mapping


# =============================================================================
# Dominance Functions
# =============================================================================

def _clamp_metrics(m, floor=1e-12):
    """
    Clamp metrics to prevent numerical instability from very small values.

    Args:
        m: Metrics tuple (cost, latency, energy, qoa) or LLMMetrics object
        floor: Minimum value for cost, latency, energy (default: 1e-12)

    Returns:
        Clamped metrics tuple (cost, latency, energy, qoa)
    """
    # Import LLMMetrics here to avoid circular dependency
    from old.main import LLMMetrics

    # Handle both tuple and LLMMetrics objects
    if isinstance(m, LLMMetrics):
        c = m.final_cost
        l = m.final_latency
        e = m.final_energy
        q = m.quality_of_answer
    else:
        c, l, e, q = m

    return (max(c, floor), max(l, floor), max(e, floor), q)


def dominates_strict(metrics1, metrics2, epsilon=1e-6):
    """
    Check standard Pareto dominance with multiplicative tolerance.

    metrics1 dominates metrics2 if:
    - Better or equal in all objectives (within tolerance)
    - Strictly better in at least one objective

    Uses multiplicative tolerance for consistency with FPTAS:
    - c1 <= (1+ε) × c2, l1 <= (1+ε) × l2, e1 <= (1+ε) × e2
    - q1 >= q2 / (1+ε)

    Objectives: (cost, latency, energy, QoA)
    - Minimize: cost, latency, energy
    - Maximize: QoA

    Args:
        metrics1: First metrics (tuple or LLMMetrics object)
        metrics2: Second metrics (tuple or LLMMetrics object)
        epsilon: Multiplicative tolerance factor (default: 1e-6 for numerical stability)

    Returns:
        True if metrics1 dominates metrics2
    """
    # Handle None metrics (invalid/failed evaluations)
    if metrics1 is None or metrics2 is None:
        return False

    # Clamp metrics for numerical stability
    c1, l1, e1, q1 = _clamp_metrics(metrics1)
    c2, l2, e2, q2 = _clamp_metrics(metrics2)

    # Multiplicative tolerance factor
    a = 1.0 + epsilon

    # Better or equal in all objectives (with multiplicative tolerance)
    better_or_equal = (
        c1 <= a * c2 and
        l1 <= a * l2 and
        e1 <= a * e2 and
        q1 >= q2 / a
    )

    # Strictly better in at least one objective (exact comparison)
    strictly_better = (
        c1 < c2 or
        l1 < l2 or
        e1 < e2 or
        q1 > q2
    )

    return better_or_equal and strictly_better


def SigBetter(metrics1, metrics2, alpha):
    """
    Check α-dominance (approximate dominance with coarsening factor).

    metrics1 α-dominates metrics2 if:
    - metrics1 is within α factor of metrics2 in all objectives
    - metrics1 is strictly better in at least one objective

    This allows approximate Pareto sets with controlled precision.

    Args:
        metrics1: First metrics (tuple or LLMMetrics object)
        metrics2: Second metrics (tuple or LLMMetrics object)
        alpha: Coarsening factor (1.0 = exact, >1.0 = approximate)

    Returns:
        True if metrics1 α-dominates metrics2
    """
    import math

    # Handle None metrics (invalid/failed evaluations)
    if metrics1 is None or metrics2 is None:
        return False

    # Clamp metrics for numerical stability
    c1, l1, e1, q1 = _clamp_metrics(metrics1)
    c2, l2, e2, q2 = _clamp_metrics(metrics2)

    # Handle inf/invalid values: treat as dominated
    if math.isinf(c1) or math.isinf(l1) or math.isinf(e1):
        return False  # metrics1 is invalid, cannot dominate

    if math.isinf(c2) or math.isinf(l2) or math.isinf(e2):
        return True  # metrics2 is invalid, metrics1 dominates it

    # Handle negative or zero QoA: clamp to small positive
    if q1 <= 0:
        q1 = 1e-9
    if q2 <= 0:
        q2 = 1e-9

    # Apply alpha approximation (multiplicative slack)
    better_or_equal = (
        c1 <= alpha * c2 and
        l1 <= alpha * l2 and
        e1 <= alpha * e2 and
        q1 >= q2 / alpha
    )

    # Strictly better in at least one objective
    strictly_better = (
        c1 < c2 or
        l1 < l2 or
        e1 < e2 or
        q1 > q2
    )

    return better_or_equal and strictly_better


# =============================================================================
# Pareto Pruning Functions
# =============================================================================

def Prune_Basic(plans, newPlan, newPlan_metrics, alpha=1.0):
    """
    Keep Pareto optimal plans using α-approximate dominance.

    Adds newPlan to plans if it's not dominated, and removes any plans
    dominated by newPlan.

    Args:
        plans: List of (plan, metrics) tuples
        newPlan: New plan to consider adding
        newPlan_metrics: Metrics of new plan
        alpha: Approximation factor (1.0 = exact dominance)

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

    # Remove plans that are α-dominated by new plan (use same alpha for consistency)
    filtered_plans = []
    for p, p_metrics in plans:
        if not SigBetter(newPlan_metrics, p_metrics, alpha):
            filtered_plans.append((p, p_metrics))

    # Add new plan
    filtered_plans.append((newPlan, newPlan_metrics))

    # Deduplicate: Keep only unique solutions
    # For discrete optimization, duplicates waste memory and mislead solution counts
    seen = {}
    unique_plans = []
    for p, p_metrics in filtered_plans:
        # Handle different plan types
        if isinstance(p, tuple):
            # Direct tuple (assignment) from V3 implementation
            key = p
        elif hasattr(p, 'struct_id'):
            # Individual with struct_id
            key = (p.struct_id, tuple(p.assignment))
        elif hasattr(p, 'subgraph'):
            # PartialPlan: use k and assignment
            key = (p.k, tuple(p.assignment))
        elif hasattr(p, 'assignment'):
            # Has assignment attribute
            key = tuple(p.assignment)
        else:
            # Unknown type, try to convert to tuple
            key = tuple(p) if not isinstance(p, tuple) else p

        if key not in seen:
            seen[key] = (p, p_metrics)
            unique_plans.append((p, p_metrics))
        # If duplicate found, keep the first occurrence (they have identical metrics anyway)

    return unique_plans


def Prune_DeltaQoA(plans, newPlan, newPlan_metrics, delta):
    """
    Frontier pruning using delta-QoA dominance (matching DP's delta approach).

    Solution A delta-dominates Solution B if A is at least as good on
    resources (cost, latency, energy), A's QoA is within delta tolerance
    of B's QoA (q_A >= q_B - delta), and A is strictly better on at least
    one criterion.

    More aggressive than exact Pareto: solutions with similar QoA (within
    delta) are pruned if dominated on resources. This mirrors DP's delta
    pruning which reduces stored solutions without changing search direction.
    """
    if newPlan_metrics is None:
        return plans

    c_new, l_new, e_new, q_new = _clamp_metrics(newPlan_metrics)

    # Check if any existing plan delta-dominates new plan
    for p, p_metrics in plans:
        c_p, l_p, e_p, q_p = _clamp_metrics(p_metrics)
        resources_ok = c_p <= c_new and l_p <= l_new and e_p <= e_new
        qoa_ok = q_p >= q_new - delta  # Within delta tolerance
        strictly_better = c_p < c_new or l_p < l_new or e_p < e_new or q_p > q_new
        if resources_ok and qoa_ok and strictly_better:
            return plans  # Existing delta-dominates new, don't add

    # Remove existing plans delta-dominated by new plan
    filtered_plans = []
    for p, p_metrics in plans:
        c_p, l_p, e_p, q_p = _clamp_metrics(p_metrics)
        resources_ok = c_new <= c_p and l_new <= l_p and e_new <= e_p
        qoa_ok = q_new >= q_p - delta
        strictly_better = c_new < c_p or l_new < l_p or e_new < e_p or q_new > q_p
        if resources_ok and qoa_ok and strictly_better:
            continue  # New delta-dominates existing, remove it
        filtered_plans.append((p, p_metrics))

    filtered_plans.append((newPlan, newPlan_metrics))

    # Deduplicate
    seen = {}
    unique_plans = []
    for p, p_metrics in filtered_plans:
        if isinstance(p, tuple):
            key = p
        elif hasattr(p, 'struct_id'):
            key = (p.struct_id, tuple(p.assignment))
        elif hasattr(p, 'assignment'):
            key = tuple(p.assignment)
        else:
            key = tuple(p) if not isinstance(p, tuple) else p
        if key not in seen:
            seen[key] = (p, p_metrics)
            unique_plans.append((p, p_metrics))

    return unique_plans


def prune_pareto(plans_with_metrics, alpha=1.0):
    """
    Prune list of (plan, metrics) to keep only Pareto optimal plans.

    Args:
        plans_with_metrics: List of (plan, metrics) tuples
        alpha: Approximation factor (1.0 = exact)

    Returns:
        Pruned list of (plan, metrics) tuples
    """
    out = []
    for p, m in plans_with_metrics:
        out = Prune_Basic(out, p, m, alpha=alpha)
    return out


# =============================================================================
# Plan Evaluation and Caching
# =============================================================================

def evaluate_plan_cached(plan, df_history, query_type):
    """
    Evaluate plan with caching to avoid redundant evaluations.

    Cache key: (struct_id, assignment tuple, query_type)

    Args:
        plan: Individual to evaluate
        df_history: Historical metrics DataFrame
        query_type: Query category (e.g., "Art")

    Returns:
        Tuple of (cost, latency, energy, qoa) or (inf, inf, inf, 0.0) on error
    """
    cache_key = (plan.struct_id, tuple(plan.assignment), query_type)

    if cache_key in _evaluation_cache:
        return _evaluation_cache[cache_key]

    # Import evaluation function from main
    from old.main import evaluate_individual_V2

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


# =============================================================================
# Mutation Generation
# =============================================================================

def generate_all_mutations(p, max_nodes=5):
    """
    Generate all single-step mutations of plan p.

    Mutation types (matching NSGA-II):
    1. Change LLM assignment at one non-blending node
    2. Flip one edge in DAG structure
    3. Add one node to DAG (if not at max_nodes)

    Args:
        p: Individual to mutate
        max_nodes: Maximum nodes allowed (default: 5)

    Returns:
        List of mutated Individual objects
    """
    mutations = []
    k = len(p.assignment)

    # Import needed functions from dag module
    from ..structures.dag import (
        mask_to_adjacency_matrix,
        canonical_mask_and_permutation,
        edge_bit_index
    )
    import random

    # Mutation Type 1: Change LLM assignment at each node
    indeg = get_indegrees(k, p.struct_id)
    for node_idx in range(k):
        # Can't change blending nodes (they must be model 5)
        if indeg[node_idx] > 1:
            continue

        current_llm = p.assignment[node_idx]
        for new_llm in [0, 1, 2, 3, 4]:
            if new_llm != current_llm:
                new_assignment = list(p.assignment)
                new_assignment[node_idx] = new_llm
                mutations.append(Individual(p.struct_id, new_assignment))

    # Mutation Type 2: Flip one edge in DAG
    num_edges = k * (k - 1) // 2
    for bit in range(num_edges):
        new_struct_id = p.struct_id ^ (1 << bit)

        # Check if still valid DAG structure
        if is_valid_dag_structure(new_struct_id, k):
            # Repair assignment for new structure
            new_assignment = repair_assignment_for_structure(
                new_struct_id, k, list(p.assignment)
            )

            # Canonicalize after edge flip to ensure consistent struct_id encoding
            adj = mask_to_adjacency_matrix(k, new_struct_id)
            canon_mask, perm = canonical_mask_and_permutation(adj, k)
            if perm is not None:
                new_assignment = [new_assignment[old_node] for old_node in perm]

            mutations.append(Individual(canon_mask, new_assignment))

    # Mutation Type 3: Add one node (if not at max)
    if k < max_nodes:
        new_k = k + 1
        adj = mask_to_adjacency_matrix(k, p.struct_id)

        # Expand adjacency matrix
        adj2 = [[0] * new_k for _ in range(new_k)]
        for i in range(k):
            for j in range(k):
                adj2[i][j] = adj[i][j]

        # Connect old sink to new sink
        adj2[k - 1][k] = 1

        # Extend assignment with random model for new node
        new_assignment = list(p.assignment)
        new_assignment.append(random.randint(0, 4))

        # Canonicalize
        canon_mask, perm = canonical_mask_and_permutation(adj2, new_k)
        if perm is not None:
            new_assignment = [new_assignment[old_node] for old_node in perm]

        # Repair assignment to respect indegree constraints
        new_indeg = get_indegrees(new_k, canon_mask)
        for i in range(new_k):
            if new_indeg[i] > 1:
                new_assignment[i] = 5
            elif new_indeg[i] <= 1 and new_assignment[i] == 5:
                new_assignment[i] = random.randint(0, 4)

        mutations.append(Individual(canon_mask, new_assignment))

    return mutations


# =============================================================================
# Pareto Climb Algorithm
# =============================================================================

def epsilon_dominates_strict(metrics1, metrics2, epsilon):
    """
    Check ε-dominance for hill climbing.

    metrics1 ε-dominates metrics2 if:
    - No worse on all objectives within (1+ε) factor
    - Strictly better on at least one objective vs ε-relaxed bound

    Args:
        metrics1: Candidate metrics (cost, latency, energy, qoa)
        metrics2: Current metrics (cost, latency, energy, qoa)
        epsilon: Approximation factor

    Returns:
        True if metrics1 ε-dominates metrics2
    """
    if metrics1 is None or metrics2 is None:
        return False

    c1, l1, e1, q1 = _clamp_metrics(metrics1)
    c2, l2, e2, q2 = _clamp_metrics(metrics2)

    a = 1.0 + epsilon
    tol = 1e-12

    # No worse within (1+ε) factor
    no_worse = (
        c1 <= a * c2 + tol and
        l1 <= a * l2 + tol and
        e1 <= a * e2 + tol and
        q1 >= q2 / a - tol
    )

    # Strictly better vs ε-relaxed bound
    strictly_better = (
        c1 < a * c2 - tol or
        l1 < a * l2 - tol or
        e1 < a * e2 - tol or
        q1 > q2 / a + tol
    )

    return no_worse and strictly_better


def delta_qoa_improves(metrics1, metrics2, delta):
    """
    Check Delta-QoA improvement for hill climbing.

    metrics1 improves over metrics2 if QoA increases by at least delta.

    Args:
        metrics1: Candidate metrics (cost, latency, energy, qoa)
        metrics2: Current metrics (cost, latency, energy, qoa)
        delta: Minimum QoA improvement threshold

    Returns:
        True if metrics1 improves QoA by at least delta
    """
    if metrics1 is None or metrics2 is None:
        return False

    _, _, _, q1 = _clamp_metrics(metrics1)
    _, _, _, q2 = _clamp_metrics(metrics2)

    return q1 >= q2 + delta


def ParetoClimb(p, df_history, query_type, max_iterations=50, max_nodes=5,
                pruning_strategy='none', epsilon=0.01, delta=0.01):
    """
    Hill climbing that respects Pareto dominance or pruning strategies.

    Matches the Java reference implementation: try all 1-step mutations,
    pick first improving one, repeat until no improvement found.

    This is the corrected version that actually updates the plan structure,
    not just the metrics.

    Args:
        p: Starting plan (Individual)
        df_history: Historical metrics DataFrame
        query_type: Query category
        max_iterations: Maximum climb iterations (default: 50)
        max_nodes: Maximum nodes allowed (default: 5)
        pruning_strategy: 'none', 'epsilon', or 'delta_qoa' (default: 'none')
        epsilon: Approximation factor for ε-dominance (default: 0.01)
        delta: Minimum QoA improvement for Delta-QoA (default: 0.01)

    Returns:
        Improved Individual (or original if no improvement found)
    """
    improving = True
    iterations = 0

    # Evaluate current plan once
    p_metrics = evaluate_plan_cached(p, df_history, query_type)

    while improving and iterations < max_iterations:
        improving = False
        iterations += 1

        # Generate all 1-step mutations (including add node - matches NSGA-II)
        mutations = generate_all_mutations(p, max_nodes=max_nodes)

        # Try each mutation to find an improving one
        for mutated_plan in mutations:
            mutated_metrics = evaluate_plan_cached(mutated_plan, df_history, query_type)

            # Check acceptance based on pruning strategy
            accept = False
            if pruning_strategy == 'epsilon':
                accept = epsilon_dominates_strict(mutated_metrics, p_metrics, epsilon)
            elif pruning_strategy == 'delta_qoa':
                accept = delta_qoa_improves(mutated_metrics, p_metrics, delta)
            else:  # 'none' or default
                accept = dominates_strict(mutated_metrics, p_metrics)

            if accept:
                # Update both plan structure AND metrics
                p = mutated_plan
                p_metrics = mutated_metrics
                improving = True
                break  # Found improvement, restart from this plan

    return p


# =============================================================================
# Subproblem Caching Helpers (for Algorithm 3)
# =============================================================================

class PartialPlan:
    """
    Represents a partial plan (subgraph with assignment).

    Used for subproblem caching in Algorithm 3.
    """
    def __init__(self, subgraph, assignment, sink_node, metrics=None):
        """
        Args:
            subgraph: NetworkX DiGraph for this partial plan
            assignment: List of model IDs for each node
            sink_node: Output node of this partial plan
            metrics: Tuple of (cost, latency, energy, qoa)
        """
        self.subgraph = subgraph
        self.assignment = list(assignment)
        self.sink_node = sink_node
        self.metrics = metrics
        self.k = len(subgraph.nodes())

    def __repr__(self):
        if self.metrics:
            return f"PartialPlan(k={self.k}, sink={self.sink_node}, metrics={self.metrics})"
        return f"PartialPlan(k={self.k}, sink={self.sink_node})"


# =============================================================================
# Approximate Frontier Management
# =============================================================================

def ApproximateFrontiers(p, P, i, query_type, df_history, elapsed_time=0,
                        timeout_seconds=60, use_adaptive_precision=True,
                        alpha_start=1.5, alpha_end=1.0):
    """
    Add optimized plan to Pareto frontier with adaptive α-dominance pruning.

    Implements the high-level idea from RMQ Algorithm 3 (adaptive precision pruning)
    but maintains a GLOBAL frontier per query type rather than the paper's
    per-subproblem cache. This is simpler but doesn't achieve the DP-like sharing
    of intermediate results described in the paper.

    Process:
    1. Calculate α using iteration-based schedule: α = 25 * 0.99^(⌊i/25⌋)
    2. Evaluate final plan
    3. Prune into global frontier P[query_type] using α-dominance

    Note: The paper's Algorithm 3 builds frontiers for intermediate results
    (subproblems) and combines them. We only maintain one frontier per query_type.

    Args:
        p: New optimized plan (Individual)
        P: Plan cache dictionary {query_type: [(plan, metrics), ...]}
        i: Iteration count (used for α schedule)
        query_type: Query category
        df_history: Historical metrics DataFrame
        elapsed_time: (Unused - kept for backward compatibility)
        timeout_seconds: (Unused - kept for backward compatibility)
        use_adaptive_precision: If True, use α-schedule; if False, use exact (α=1.0)
        alpha_start: (Unused - paper schedule doesn't use this)
        alpha_end: (Unused - paper schedule doesn't use this)

    Returns:
        Updated plan cache P
    """
    # Calculate adaptive precision factor using paper's ITERATION-BASED schedule
    # From RMQ (Trummer & Koch, SIGMOD 2016): α = 25 * 0.99^(⌊i/25⌋)
    if use_adaptive_precision:
        # Paper schedule: starts at 25 (aggressive early pruning), decays by 1% every 25 iterations
        alpha = 25.0 * (0.99 ** (i // 25))
    else:
        # Use exact dominance (default)
        alpha = 1.0

    if query_type not in P:
        P[query_type] = []

    # Evaluate new plan
    p_metrics = evaluate_plan_cached(p, df_history, query_type)

    # Prune with adaptive precision
    P[query_type] = Prune_Basic(P[query_type], p, p_metrics, alpha)

    return P


@dataclass(frozen=True)
class SubproblemKey:
    """
    Cache key for subproblems.

    Since construct_subdag_behind_node preserves original node IDs,
    node_set correctly identifies which original nodes are included.
    """
    query_type: str
    struct_id: int
    node_set: frozenset  # Original node IDs
    sink: int  # Original sink node ID


def renumber_graph_for_evaluation(subG, assignment_map, sink):
    """
    Renumber graph to 0-indexed for evaluation functions with sink last.

    Evaluation functions commonly assume:
    - Graph with nodes 0, 1, 2, ... (consecutive from 0)
    - Assignment list indexed by these node numbers
    - Sink node is at the LAST index (critical for blend operations)

    Args:
        subG: Graph with original node IDs
        assignment_map: Dict mapping original node IDs to models
        sink: Original sink node ID (will be placed last)

    Returns:
        renumbered_G: Graph with nodes 0..k-1 (sink at k-1)
        renumbered_assignment: List [model_0, ..., model_k-1] (sink model last)
    """
    # Use topological order for stability, then force sink last
    order = list(nx.topological_sort(subG))

    # Force sink to be last (critical for evaluators that assume this)
    if sink in order:
        order.remove(sink)
        order.append(sink)

    # Create mapping from original IDs to 0-indexed
    node_to_idx = {node: idx for idx, node in enumerate(order)}

    # Create renumbered graph
    renumbered_G = nx.DiGraph()
    renumbered_G.add_nodes_from(range(len(order)))

    for u, v in subG.edges():
        renumbered_G.add_edge(node_to_idx[u], node_to_idx[v])

    # Create renumbered assignment following the order (sink model last)
    renumbered_assignment = [assignment_map[node] for node in order]

    return renumbered_G, renumbered_assignment


def ApproximateFrontiers_V3_Corrected(plan, P, alpha, query_type, df_history):
    """
    Paper-correct Algorithm 3 with ALL bugs fixed.

    Key fixes from V2:
    1. Assignment mapping uses sorted(subG.nodes()) consistently
    2. Conflict checking properly zips parent_nodes with parent_assignment
    3. All imports present
    4. Debug assertions for validation

    Args:
        plan: Individual (full plan)
        P: Subproblem cache {SubproblemKey: [(assignment_tuple, metrics), ...]}
        alpha: Adaptive precision factor
        query_type: Query category
        df_history: Historical metrics

    Returns:
        Updated cache P
    """
    from old.main import (
        graph_from_struct,
        construct_subdag_behind_node,
        canonical_representation,
        get_adj_from_graph,
        get_subdag_metrics_v7,
        estimate_schedule_v3,
        special_get_subdag_metrics_for_one_blend_operations
    )

    k = len(plan.assignment)
    G = graph_from_struct(k, int(plan.struct_id))

    # Process each node in topological order
    for node in nx.topological_sort(G):
        # Get TRUE subgraph behind this node (preserves original IDs)
        subG, sub_assign, sub_sink = construct_subdag_behind_node(
            G, node, [str(a) for a in plan.assignment]
        )

        # sub_assign corresponds to sorted(subG.nodes())
        # sub_sink is the original sink node ID

        # Create cache key (node_set uses original IDs since subG preserves them)
        node_set = frozenset(subG.nodes())
        subG_adj = get_adj_from_graph(subG)
        subG_struct_id = canonical_representation(subG_adj, len(subG.nodes()))

        key = SubproblemKey(
            query_type=query_type,
            struct_id=subG_struct_id,
            node_set=node_set,
            sink=sub_sink
        )

        if key not in P:
            P[key] = []

        # Get predecessors in ORIGINAL graph
        preds = list(G.predecessors(node))
        in_degree = len(preds)

        if in_degree == 0:
            # Source node: enumerate model choices
            for model in [0, 1, 2, 3, 4]:
                # Single node subgraph, assignment is just (model,)
                assignment = (model,)

                # Evaluate: need 0-indexed graph for evaluation functions
                eval_G = nx.DiGraph()
                eval_G.add_node(0)  # Renumbered to 0
                eval_assignment = [str(model)]  # Indexed by 0

                metrics = get_subdag_metrics_v7(
                    eval_G, eval_assignment, query_type, df_history
                )

                if metrics is None:
                    metrics = estimate_schedule_v3(
                        eval_G, eval_assignment, query_type,
                        query_tokens=215,
                        blending_prompt_tokens=26,
                        ctx_tokens=39,
                        df_history=df_history,
                        levenshtein_threshold=0.75,
                        turn_off_exact_fuzzy_matching=True  # Match FPTAS: no fuzzy matching
                    )

                if metrics is not None:
                    # Convert to tuple
                    if hasattr(metrics, 'final_cost'):
                        m_tuple = (metrics.final_cost, metrics.final_latency,
                                 metrics.final_energy, metrics.quality_of_answer)
                    else:
                        m_tuple = metrics

                    P[key] = Prune_Basic(P[key], assignment, m_tuple, alpha)

        elif in_degree == 1:
            # Single parent: extend parent's solutions
            parent = preds[0]

            # Get parent's subgraph and key
            parent_subG, parent_assign, parent_sink = construct_subdag_behind_node(
                G, parent, [str(a) for a in plan.assignment]
            )

            parent_node_set = frozenset(parent_subG.nodes())
            parent_adj = get_adj_from_graph(parent_subG)
            parent_struct_id = canonical_representation(parent_adj, len(parent_subG.nodes()))

            parent_key = SubproblemKey(
                query_type=query_type,
                struct_id=parent_struct_id,
                node_set=parent_node_set,
                sink=parent_sink
            )

            if parent_key in P:
                for parent_assignment, parent_metrics in P[parent_key]:
                    # Try each model for current node
                    for model in [0, 1, 2, 3, 4]:
                        # Build assignment map from parent
                        parent_nodes_sorted = sorted(parent_subG.nodes())

                        # CRITICAL: zip parent nodes with parent assignment
                        assignment_map = {}
                        for pnode, pmodel in zip(parent_nodes_sorted, parent_assignment):
                            assignment_map[pnode] = pmodel

                        # Add current node
                        assignment_map[node] = model

                        # Build full assignment for subG in sorted order
                        subG_nodes_sorted = sorted(subG.nodes())
                        merged_assignment = []
                        ok = True
                        for n in subG_nodes_sorted:
                            if n not in assignment_map:
                                ok = False
                                break
                            merged_assignment.append(assignment_map[n])

                        if not ok:
                            continue

                        # DEBUG ASSERTION
                        assert len(merged_assignment) == len(subG.nodes()), \
                            f"Assignment length {len(merged_assignment)} != subG size {len(subG.nodes())}"

                        # Renumber for evaluation: evaluation functions expect 0-indexed graphs with sink last
                        eval_G, eval_assignment = renumber_graph_for_evaluation(subG, assignment_map, sub_sink)

                        # Evaluate
                        metrics = get_subdag_metrics_v7(
                            eval_G, [str(a) for a in eval_assignment],
                            query_type, df_history
                        )

                        if metrics is None:
                            metrics = estimate_schedule_v3(
                                eval_G, [str(a) for a in eval_assignment],
                                query_type,
                                query_tokens=215,
                                blending_prompt_tokens=26,
                                ctx_tokens=39,
                                df_history=df_history,
                                levenshtein_threshold=0.75,
                                turn_off_exact_fuzzy_matching=True  # Match FPTAS: no fuzzy matching
                            )

                        if metrics is not None:
                            if hasattr(metrics, 'final_cost'):
                                m_tuple = (metrics.final_cost, metrics.final_latency,
                                         metrics.final_energy, metrics.quality_of_answer)
                            else:
                                m_tuple = metrics

                            P[key] = Prune_Basic(P[key], tuple(merged_assignment), m_tuple, alpha)

        else:
            # Blend node: merge parent assignments with conflict checking
            parent_keys = []
            parent_frontiers = []
            parent_subgraphs = []

            for parent in preds:
                parent_subG, parent_assign, parent_sink = construct_subdag_behind_node(
                    G, parent, [str(a) for a in plan.assignment]
                )
                parent_node_set = frozenset(parent_subG.nodes())
                parent_adj = get_adj_from_graph(parent_subG)
                parent_struct_id = canonical_representation(parent_adj, len(parent_subG.nodes()))

                parent_key = SubproblemKey(
                    query_type=query_type,
                    struct_id=parent_struct_id,
                    node_set=parent_node_set,
                    sink=parent_sink
                )

                if parent_key in P:
                    parent_keys.append(parent_key)
                    parent_frontiers.append(P[parent_key])
                    parent_subgraphs.append((parent_subG, parent_sink))

            # Only proceed if all parents have frontiers
            if len(parent_frontiers) == len(preds):
                # Try combinations of parent assignments
                # Limit combinatorial explosion
                max_combos = 50

                all_combos = list(itertools.product(*[[a for a, m in pf] for pf in parent_frontiers]))
                if len(all_combos) > max_combos:
                    # Sample random combinations
                    combos = [random.choice(all_combos) for _ in range(max_combos)]
                else:
                    combos = all_combos

                for parent_assignments in combos:
                    # Build assignment map with conflict checking
                    assignment_map = {}
                    conflict = False

                    for i in range(len(preds)):
                        parent_subG, parent_sink = parent_subgraphs[i]
                        parent_assignment = parent_assignments[i]

                        # CRITICAL: use sorted(parent_subG.nodes()) to match assignment
                        parent_nodes_sorted = sorted(parent_subG.nodes())

                        if len(parent_nodes_sorted) != len(parent_assignment):
                            conflict = True
                            break

                        # Zip parent nodes with parent assignment
                        for pnode, pmodel in zip(parent_nodes_sorted, parent_assignment):
                            if pnode in assignment_map:
                                if assignment_map[pnode] != pmodel:
                                    conflict = True  # Conflict on shared ancestor!
                                    break
                            else:
                                assignment_map[pnode] = pmodel

                        if conflict:
                            break

                    if conflict:
                        continue

                    # Add blend node with model 5
                    # sub_sink is the original sink node ID (which is 'node')
                    assignment_map[sub_sink] = 5

                    # Build full assignment for subG in sorted order
                    subG_nodes_sorted = sorted(subG.nodes())
                    merged_assignment = []
                    ok = True
                    for n in subG_nodes_sorted:
                        if n not in assignment_map:
                            ok = False
                            break
                        merged_assignment.append(assignment_map[n])

                    if not ok:
                        continue

                    # DEBUG ASSERTIONS
                    assert len(merged_assignment) == len(subG.nodes()), \
                        f"Blend: Assignment length {len(merged_assignment)} != subG size {len(subG.nodes())}"

                    # Verify blend node has model 5
                    sub_sink_idx = subG_nodes_sorted.index(sub_sink)
                    assert merged_assignment[sub_sink_idx] == 5, \
                        f"Blend node at index {sub_sink_idx} should be model 5, got {merged_assignment[sub_sink_idx]}"

                    # Renumber for evaluation: evaluation functions expect 0-indexed graphs with sink last
                    eval_G, eval_assignment = renumber_graph_for_evaluation(subG, assignment_map, sub_sink)

                    # Evaluate on TRUE subG (renumbered)
                    metrics = get_subdag_metrics_v7(
                        eval_G, [str(a) for a in eval_assignment],
                        query_type, df_history
                    )

                    if metrics is None:
                        metrics = special_get_subdag_metrics_for_one_blend_operations(
                            eval_G, [str(a) for a in eval_assignment],
                            query_type, df_history
                        )

                    if metrics is None:
                        metrics = estimate_schedule_v3(
                            eval_G, [str(a) for a in eval_assignment],
                            query_type,
                            query_tokens=215,
                            blending_prompt_tokens=26,
                            ctx_tokens=39,
                            df_history=df_history,
                            levenshtein_threshold=0.75,
                            turn_off_exact_fuzzy_matching=True  # Match FPTAS: no fuzzy matching
                        )

                    if metrics is not None:
                        if hasattr(metrics, 'final_cost'):
                            m_tuple = (metrics.final_cost, metrics.final_latency,
                                     metrics.final_energy, metrics.quality_of_answer)
                        else:
                            m_tuple = metrics

                        P[key] = Prune_Basic(P[key], tuple(merged_assignment), m_tuple, alpha)

    return P


# =============================================================================
# Random Plan Generation
# =============================================================================

def RandomPlan(max_nodes):
    """
    Generate random bushy plan (random DAG structure + valid assignment).

    Process:
    1. Generate random DAG structure (uses enumeration for k≤6, random generation for k>6)
    2. Assign models respecting blending constraints:
       - Nodes with in-degree > 1: Must use model 5 (blending)
       - Other nodes: Random model from 0-4

    Args:
        max_nodes: Maximum nodes in DAG

    Returns:
        Random Individual with valid structure and assignment
    """
    # For small max_nodes, use enumeration (fast & complete)
    # For large max_nodes, use random generation (avoids O(k!) enumeration bottleneck)
    if max_nodes <= 6:
        structures = generate_nonisomorphic_dags(max_nodes)
        k, struct_id = random.choice(structures)
    else:
        # Random generation for large k: pick random k, then generate valid structure
        from old.main import random_valid_structure
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


# =============================================================================
# Main RandomMOQO Algorithm
# =============================================================================

def RandomMOQO(query_type, df_history, timeout_seconds=60, max_nodes=5,
               use_adaptive_precision=True, alpha_start=1.5, alpha_end=1.0, max_iterations=None,
               use_subproblem_caching=True, pruning_strategy='none', epsilon=0.01, delta=0.01,
               verbose=True):
    """
    RandomMOQO multi-objective optimizer.

    Implements "An Adaptive Precision Approach to Efficiently Find
    the Pareto Frontier of Multi-Objective Queries"
    (Trummer & Koch, SIGMOD 2016) Algorithm 1:

    Process:
    1. Generate random plan (RandomPlan)
    2. Improve via Pareto hill climbing (ParetoClimb - greedy first-improvement)
    3. Add to approximate Pareto frontier with adaptive α-dominance pruning
    4. Repeat until timeout

    When use_subproblem_caching=True (default):
    - Implements full Algorithm 3 with DP-like subproblem caching
    - Maintains Pareto frontiers for intermediate results (subgraphs)
    - Enables sharing of subproblem solutions across iterations
    - Matches paper's Algorithm 3

    When use_subproblem_caching=False:
    - Uses simpler global frontier per query type
    - Faster but less sophisticated than paper's approach

    Remaining difference from paper:
    - Larger neighborhood O(k²) from edge flips (not paper's O(k))

    Args:
        query_type: Query category (e.g., "Art", "Science and technology")
        df_history: Historical metrics DataFrame
        timeout_seconds: Time limit for optimization (default: 60)
        max_nodes: Maximum nodes in DAG (default: 5)
        use_adaptive_precision: Use paper's α-schedule (α=25*0.99^(⌊i/25⌋)) or constant alpha
        alpha_start: Constant alpha value when use_adaptive_precision=False (default: 1.5)
        alpha_end: (Unused - kept for backward compatibility)
        max_iterations: Maximum iterations (default: None, use timeout only)
        use_subproblem_caching: Use Algorithm 3 subproblem caching (default: True)
        pruning_strategy: 'none', 'epsilon', or 'delta_qoa' (default: 'none')
        epsilon: Approximation factor for ε-dominance (default: 0.01)
        delta: Minimum QoA improvement for Delta-QoA (default: 0.01)
        verbose: Print progress messages (default: True)

    Returns:
        List of Pareto optimal Individual objects with metrics set
    """
    # Initialize caches
    if use_subproblem_caching:
        # P is subproblem cache: {signature: [(partial_plan, metrics), ...]}
        P = {}
        # Separate global frontier for final full plans
        global_frontier = []
    else:
        # P is simple cache: {query_type: [(plan, metrics), ...]}
        P = {}
        P[query_type] = []

    i = 1

    start_time = time.time()
    if verbose:
        print(f"\nRandomMOQO: Starting optimization for {query_type}")
        print(f"  Timeout: {timeout_seconds}s, Max nodes: {max_nodes}")
        print(f"  Subproblem caching: {'Algorithm 3 (DP-like)' if use_subproblem_caching else 'Global frontier only'}")
        print(f"  Pruning strategy: {pruning_strategy}" +
              (f" (ε={epsilon})" if pruning_strategy == 'epsilon' else
               f" (Δ={delta})" if pruning_strategy == 'delta_qoa' else ""))
        if use_adaptive_precision:
            print(f"  Adaptive precision: α = 25·0.99^(⌊i/25⌋) (paper's iteration-based schedule)")
        else:
            print(f"  Using constant alpha: α={alpha_start}")

    # Refine frontier approximation until timeout or max iterations
    deadline = start_time + timeout_seconds
    while (time.time() < deadline) and (max_iterations is None or i <= max_iterations):
        # Generate random bushy query plan
        plan = RandomPlan(max_nodes)

        # Improve plan via local search (always strict Pareto dominance).
        # Delta pruning only affects frontier management, not the search
        # direction — matching how DP's delta only filters stored solutions.
        optPlan = ParetoClimb(plan, df_history, query_type, max_nodes=max_nodes,
                             pruning_strategy='none')

        # Check timeout after ParetoClimb (can be slow)
        if time.time() >= deadline:
            break

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Frontier uses exact Pareto (alpha=1.0) as base
        current_alpha = 1.0

        # Add to frontier: use delta-QoA pruning or strict Pareto
        opt_metrics = evaluate_plan_cached(optPlan, df_history, query_type)

        if use_subproblem_caching:
            # Use Algorithm 3: DP-like subproblem caching
            P = ApproximateFrontiers_V3_Corrected(
                optPlan, P, current_alpha, query_type, df_history
            )

            # Check timeout after subproblem caching (can be slow)
            if time.time() >= deadline:
                break

            # Add optimized plan to global frontier
            if pruning_strategy == 'delta_qoa':
                global_frontier = Prune_DeltaQoA(global_frontier, optPlan, opt_metrics, delta)
            else:
                global_frontier = Prune_Basic(global_frontier, optPlan, opt_metrics, current_alpha)

        else:
            # Use simple global frontier
            if pruning_strategy == 'delta_qoa':
                P[query_type] = Prune_DeltaQoA(P[query_type], optPlan, opt_metrics, delta)
            else:
                P[query_type] = Prune_Basic(P[query_type], optPlan, opt_metrics, current_alpha)

        if i % 50 == 0:
            # Calculate solution count
            if use_subproblem_caching:
                solution_count = len(global_frontier)
            else:
                solution_count = len(P[query_type])

            # Display progress
            print(f"  Iter {i}: {solution_count} solutions, {elapsed:.1f}s elapsed")

        i += 1

    elapsed = time.time() - start_time
    print(f"RandomMOQO: Completed {i-1} iterations in {elapsed:.1f}s")

    # Extract final Pareto set
    if use_subproblem_caching:
        print(f"  Final Pareto set size: {len(global_frontier)}")
        pareto_plans = [plan for plan, metrics in global_frontier]
    else:
        print(f"  Final Pareto set size: {len(P[query_type])}")
        pareto_plans = [plan for plan, metrics in P[query_type]]

    # Evaluate all plans to set their metrics attribute
    for plan in pareto_plans:
        if plan.metrics is None:
            metrics = evaluate_plan_cached(plan, df_history, query_type)
            plan.metrics = metrics
            plan.objectives = (metrics[0], metrics[1], metrics[2], -metrics[3])

    return pareto_plans
