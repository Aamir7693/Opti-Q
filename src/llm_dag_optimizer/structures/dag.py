"""
DAG Utilities Module

Contains all DAG structure operations:
- Edge bit indexing and adjacency matrix conversions
- Canonical representation for isomorphism detection
- DAG generation and validation
- Structure manipulation utilities
"""

import itertools
import random
from typing import List, Tuple
from functools import lru_cache


# =============================================================================
# Edge and Adjacency Matrix Operations
# =============================================================================

def edge_bit_index(i: int, j: int, k: int) -> int:
    """
    Calculate bit index for edge (i,j) in k-node DAG.

    Bit ordering: (0,1),(0,2)...(0,k-1),(1,2)...(k-2,k-1)
    Requires i < j.

    Args:
        i: Source node (must be < j)
        j: Destination node (must be > i)
        k: Total number of nodes

    Returns:
        Bit index for this edge
    """
    if not (0 <= i < j < k):
        raise ValueError(f"edge_bit_index requires 0 <= i < j < k, got i={i}, j={j}, k={k}")
    idx = 0
    for a in range(i):
        idx += (k - 1 - a)
    idx += (j - i - 1)
    return idx


def mask_to_adjacency_matrix(k: int, mask: int) -> List[List[int]]:
    """
    Convert bitmask to adjacency matrix.

    Args:
        k: Number of nodes
        mask: Bitmask representing edges

    Returns:
        k×k adjacency matrix
    """
    adj = [[0] * k for _ in range(k)]
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if mask & (1 << bit_index):
                adj[i][j] = 1
            bit_index += 1
    return adj


def adjacency_matrix_from_mask(mask: int, k: int) -> List[List[int]]:
    """
    Alternative name for mask_to_adjacency_matrix.
    Kept for backward compatibility.
    """
    return mask_to_adjacency_matrix(k, mask)


# Alias for consistency with existing code
mask_to_adj = mask_to_adjacency_matrix


# =============================================================================
# Canonical Representation (Isomorphism Detection)
# =============================================================================

def canonical_mask_and_permutation(adj_matrix, k: int) -> Tuple[int, Tuple]:
    """
    Find canonical (highest-value) bitmask representation and the permutation.

    Returns:
        (best_mask, best_perm) where best_perm[new_pos] = old_node_id
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


# Alias for consistency
canonical_mask_and_perm = canonical_mask_and_permutation


def canonical_representation_compute(adj_matrix, k: int) -> int:
    """
    Compute highest-value adjacency bitmask among all isomorphic labelings.

    This is the core function for detecting graph isomorphism.
    """
    nodes = list(range(k))
    best_mask = -1

    for perm in itertools.permutations(nodes):
        # Check if perm is valid topological order
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
                x = perm[a]
                y = perm[b]
                if adj_matrix[x][y] == 1:
                    new_mask |= (1 << bit_index)
                bit_index += 1
        if new_mask > best_mask:
            best_mask = new_mask

    return best_mask


# Cache canonical representation for performance
@lru_cache(maxsize=20000)
def canonical_representation_cached(adj_tuple, k: int) -> int:
    """Cached version of canonical_representation."""
    adj = [list(row) for row in adj_tuple]
    return canonical_representation_compute(adj, k)


def canonical_representation(adj_matrix, k: int) -> int:
    """
    Public interface for canonical representation with caching.
    """
    # Convert to tuple for hashing
    adj_tuple = tuple(tuple(int(v) for v in row) for row in adj_matrix)
    return canonical_representation_cached(adj_tuple, int(k))


# =============================================================================
# DAG Generation
# =============================================================================

@lru_cache(maxsize=16)
def generate_nonisomorphic_dags(max_nodes: int) -> List[Tuple[int, int]]:
    """
    Generate all non-isomorphic fully-connected DAGs with single sink.

    Returns:
        List of (num_nodes, structure_id) tuples
    """
    unique_dags = []
    seen_structs = set()

    for k in range(1, max_nodes + 1):
        num_edges = k * (k - 1) // 2
        for mask in range(1 << num_edges):
            # Skip empty graph for k>1
            if k > 1 and mask == 0:
                continue

            # Reconstruct adjacency matrix from bitmask
            adj = mask_to_adjacency_matrix(k, mask)

            # Check one-sink condition: only highest-index node has out-degree 0
            outdeg = [0] * k
            indeg = [0] * k
            for i in range(k):
                for j in range(k):
                    if adj[i][j] == 1:
                        outdeg[i] += 1
                        indeg[j] += 1

            # Ensure no other node besides sink has out-degree 0
            if any(outdeg[node] == 0 for node in range(k - 1)):
                continue

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
            if not all(reachable):
                continue

            # Compute canonical structure id
            canon_mask = canonical_representation(adj, k)
            if (k, canon_mask) not in seen_structs:
                seen_structs.add((k, canon_mask))
                unique_dags.append((k, canon_mask))

    return unique_dags


# =============================================================================
# DAG Validation
# =============================================================================

def is_valid_dag_structure(struct_id: int, k: int) -> bool:
    """
    Check if structure is valid DAG.

    Validation:
    - Single sink (only last node has out-degree 0)
    - All other nodes have out-degree > 0

    Args:
        struct_id: Structure bitmask
        k: Number of nodes

    Returns:
        True if valid DAG structure
    """
    adj = mask_to_adjacency_matrix(k, struct_id)

    # Check single sink (only last node has out-degree 0)
    outdeg = [sum(adj[i]) for i in range(k)]
    if any(outdeg[i] == 0 for i in range(k-1)):
        return False
    if outdeg[k-1] != 0:
        return False

    return True


def get_indegrees(k: int, struct_id: int) -> List[int]:
    """
    Calculate in-degree for each node in DAG.

    Args:
        k: Number of nodes
        struct_id: Structure bitmask

    Returns:
        List of in-degrees (one per node)
    """
    indeg = [0] * k
    bit_index = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            if struct_id & (1 << bit_index):
                indeg[j] += 1
            bit_index += 1
    return indeg


# =============================================================================
# Structure Assignment Repair
# =============================================================================

def repair_assignment_for_structure(struct_id: int, k: int, assignment: List[int]) -> List[int]:
    """
    Repair LLM assignment to match structure's in-degree constraints.

    Rules:
    - Nodes with in-degree > 1 must be blending (model 5)
    - Nodes with in-degree <= 1 must NOT be blending (models 0-4)

    Args:
        struct_id: Structure bitmask
        k: Number of nodes
        assignment: Current LLM assignment

    Returns:
        Repaired assignment
    """
    indeg = get_indegrees(k, struct_id)
    new_assignment = list(assignment)

    for i in range(k):
        if indeg[i] > 1 and new_assignment[i] != 5:
            new_assignment[i] = 5  # Must be blending
        elif indeg[i] <= 1 and new_assignment[i] == 5:
            # Was blending but no longer needed
            new_assignment[i] = random.randint(0, 4)

    return new_assignment


def assign_models_to_dag(k: int, struct_id: int) -> List[int]:
    """
    Generate valid random LLM assignment for DAG structure.

    Args:
        k: Number of nodes
        struct_id: Structure bitmask

    Returns:
        Valid LLM assignment
    """
    indeg = get_indegrees(k, struct_id)
    assignment = []
    for node in range(k):
        if indeg[node] > 1:
            assignment.append(5)  # Blending model for merge nodes
        else:
            assignment.append(random.randint(0, 4))  # Random base LLM
    return assignment
