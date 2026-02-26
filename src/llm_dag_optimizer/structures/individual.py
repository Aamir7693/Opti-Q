"""
Individual Module

Defines the Individual class and genetic operators:
- Individual representation (structure + assignment)
- Mutation operators
- Crossover operators
- Population initialization
"""

import random
import copy
from typing import List, Tuple
from .dag import (
    get_indegrees,
    mask_to_adjacency_matrix,
    canonical_mask_and_permutation,
    edge_bit_index,
    generate_nonisomorphic_dags
)


class Individual:
    """
    Represents a solution: DAG structure + LLM assignment.

    Attributes:
        struct_id: DAG structure identifier (canonical bitmask)
        assignment: Tuple of LLM model indices per node
        metrics: (cost, latency, energy, qoa)
        objectives: (cost, latency, energy, -qoa) for minimization
        rank: Pareto front rank (0 = non-dominated)
        crowding_distance: NSGA-II crowding distance metric
    """

    def __init__(self, struct_id: int, assignment: List[int]):
        self.struct_id = struct_id
        self.assignment = tuple(assignment)
        self.metrics = None
        self.objectives = None
        self.rank = None
        self.crowding_distance = None


# =============================================================================
# Population Initialization
# =============================================================================

def initialize_population(pop_size: int, max_nodes: int) -> List[Individual]:
    """
    Initialize population with random valid DAGs and assignments.

    Args:
        pop_size: Population size
        max_nodes: Maximum nodes in DAG

    Returns:
        List of Individual objects
    """
    structures = generate_nonisomorphic_dags(max_nodes)
    population = []

    for _ in range(pop_size):
        k, struct_id = random.choice(structures)
        indeg = get_indegrees(k, struct_id)

        # Random LLM assignment (respecting blending constraints)
        assignment = [None] * k
        for i in range(k):
            assignment[i] = 5 if indeg[i] > 1 else random.randint(0, 4)

        individual = Individual(struct_id, assignment)
        population.append(individual)

    return population


# =============================================================================
# Crossover Operator
# =============================================================================

def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Single-point crossover on both structure and assignment.

    If parents have different sizes, randomly select one parent.
    Otherwise, performs crossover on both structure bits and assignment.

    Args:
        parent1: First parent
        parent2: Second parent

    Returns:
        Offspring individual
    """
    if len(parent1.assignment) != len(parent2.assignment):
        return copy.deepcopy(parent1 if random.random() < 0.5 else parent2)

    k = len(parent1.assignment)
    num_bits = k * (k - 1) // 2

    mask1, mask2 = int(parent1.struct_id), int(parent2.struct_id)

    # Crossover on structure
    if num_bits > 1:
        cp = random.randint(1, num_bits - 1)
        new_mask = ((mask1 & ((1 << cp) - 1)) | (mask2 & ~((1 << cp) - 1)))
    else:
        new_mask = mask1 if random.random() < 0.5 else mask2

    # Crossover on assignment
    assign1, assign2 = list(parent1.assignment), list(parent2.assignment)
    if k > 1:
        cp2 = random.randint(1, k - 1)
        new_assign = assign1[:cp2] + assign2[cp2:]
    else:
        new_assign = assign1[:]

    # Repair: ensure only last node is sink
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

    # Repair assignments for indegree constraints
    indeg = get_indegrees(k, new_mask)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)

    # Canonicalize AND permute assignment
    adj = mask_to_adjacency_matrix(k, new_mask)
    canon_mask, perm = canonical_mask_and_permutation(adj, k)

    if perm is not None:
        new_assign = [new_assign[old_node] for old_node in perm]

    # Enforce blend constraints after canonicalization
    indeg = get_indegrees(k, canon_mask)
    for i in range(k):
        if indeg[i] > 1:
            new_assign[i] = 5
        elif indeg[i] <= 1 and new_assign[i] == 5:
            new_assign[i] = random.randint(0, 4)

    return Individual(canon_mask, new_assign)


# =============================================================================
# Mutation Operator
# =============================================================================

def mutate(ind: Individual, max_nodes: int,
           add_node_prob: float = 0.1,
           flip_edge_prob: float = 0.3,
           model_mutation_prob: float = 0.2) -> Individual:
    """
    Mutation operator with three types:
    1. Add node (if under max_nodes)
    2. Flip edge bit
    3. Change LLM assignment

    Args:
        ind: Individual to mutate
        max_nodes: Maximum nodes allowed
        add_node_prob: Probability of adding a node
        flip_edge_prob: Probability of flipping an edge
        model_mutation_prob: Probability of mutating model assignment

    Returns:
        Mutated individual
    """
    k = len(ind.assignment)
    struct_id = int(ind.struct_id)
    assign = list(ind.assignment)

    # Mutation Type 1: Add node
    if k < max_nodes and random.random() < add_node_prob:
        new_k = k + 1
        adj = mask_to_adjacency_matrix(k, struct_id)

        # Expand adjacency matrix
        adj2 = [[0]*new_k for _ in range(new_k)]
        for i in range(k):
            for j in range(k):
                adj2[i][j] = adj[i][j]

        # Connect old sink to new sink
        adj2[k-1][k] = 1

        assign.append(random.randint(0, 4))

        canon_mask, perm = canonical_mask_and_permutation(adj2, new_k)
        if perm is not None:
            assign = [assign[old_node] for old_node in perm]

        indeg = get_indegrees(new_k, canon_mask)
        for i in range(new_k):
            if indeg[i] > 1:
                assign[i] = 5
            elif indeg[i] <= 1 and assign[i] == 5:
                assign[i] = random.randint(0, 4)

        return Individual(canon_mask, assign)

    # Mutation Type 2: Flip an edge
    if k > 1 and random.random() < flip_edge_prob:
        num_bits = k * (k - 1) // 2
        bit_to_flip = random.randrange(num_bits)
        struct_id ^= (1 << bit_to_flip)

    # Mutation Type 3: Mutate base model nodes
    for i in range(k):
        if assign[i] != 5 and random.random() < model_mutation_prob:
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
    adj = mask_to_adjacency_matrix(k, struct_id)
    canon_mask, perm = canonical_mask_and_permutation(adj, k)
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


# =============================================================================
# Genotype Operations
# =============================================================================

def genotype_key(ind: Individual) -> Tuple[int, Tuple[int, ...]]:
    """
    Get unique key for genotype (structure + assignment).

    Args:
        ind: Individual

    Returns:
        (struct_id, assignment) tuple for hashing
    """
    return (int(ind.struct_id), tuple(ind.assignment))


def enforce_genotype_dedup(population: List[Individual]) -> List[Individual]:
    """
    Remove duplicate genotypes from population.

    Args:
        population: List of individuals

    Returns:
        Deduplicated population
    """
    seen = set()
    new_pop = []
    for ind in population:
        k = genotype_key(ind)
        if k not in seen:
            seen.add(k)
            new_pop.append(ind)
    return new_pop
