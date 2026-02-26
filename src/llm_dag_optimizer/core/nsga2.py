"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.

This module implements the NSGA-II algorithm for optimizing LLM execution plans.
Currently imports from main.py - future versions will have full implementation here.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add parent directory to import main.py
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import NSGA-II functions from main.py
try:
    from old.main import (
        nsga2_optimize as _nsga2_optimize_impl,
        fast_non_dominated_sort,
        assign_crowding_distance,
        tournament_selection,
        deduplicate_population,
    )
except ImportError as e:
    print(f"Warning: Could not import NSGA-II from main.py: {e}")
    def _nsga2_optimize_impl(*args, **kwargs):
        raise NotImplementedError("NSGA-II not available")


def run_nsga2(
    query_type: str,
    df_history: pd.DataFrame,
    pop_size: int = 200,
    generations: int = 200,
    max_nodes: int = 5,
    query_tokens: int = 215,
    blending_prompt_tokens: int = 26,
    ctx_tokens: int = 39,
    verbose: bool = True,
) -> List:
    """
    Run NSGA-II algorithm for multi-objective LLM plan optimization.

    Args:
        query_type: Query category (e.g., "Art", "Science and technology")
        df_history: Historical performance DataFrame
        pop_size: Population size (default: 200)
        generations: Number of generations (default: 200)
        max_nodes: Maximum nodes in DAG (default: 5)
        query_tokens: Query token count (default: 215)
        blending_prompt_tokens: Blending prompt tokens (default: 26)
        ctx_tokens: Context tokens (default: 39)
        verbose: Print progress (default: True)

    Returns:
        List of Pareto-optimal solutions (rank 0)

    Algorithm:
        1. Initialize random population
        2. For each generation:
           a. Evaluate fitness
           b. Fast non-dominated sort
           c. Calculate crowding distance
           d. Tournament selection
           e. Crossover and mutation
           f. Combine parent + offspring
           g. Select next generation
        3. Return Pareto front

    Components:
        - Fast non-dominated sorting: O(MN²) where M=objectives, N=population
        - Crowding distance: Diversity preservation
        - Binary tournament: Parent selection
        - Genetic operators: Crossover, mutation
    """
    if verbose:
        print("="*80)
        print("NSGA-II Algorithm")
        print("="*80)
        print(f"Query type: {query_type}")
        print(f"Population size: {pop_size}")
        print(f"Generations: {generations}")
        print(f"Max nodes: {max_nodes}")
        print("="*80)
        print()

    # Call the main implementation
    pareto_front = _nsga2_optimize_impl(
        query_tokens=query_tokens,
        blending_prompt_tokens=blending_prompt_tokens,
        ctx_tokens=ctx_tokens,
        df_history=df_history,
        pop_size=pop_size,
        generations=generations,
        max_nodes=max_nodes,
        query_type=query_type,
        verbose=verbose
    )

    if verbose:
        print(f"\n✓ NSGA-II completed")
        print(f"  Pareto front size: {len(pareto_front)}")
        if pareto_front:
            # Handle both old Individual (with .metrics) and new Individual (with .qoa, .cost)
            def get_qoa(ind):
                return ind.metrics[3] if hasattr(ind, 'metrics') else ind.qoa
            def get_cost(ind):
                return ind.metrics[0] if hasattr(ind, 'metrics') else ind.cost

            print(f"  QoA range: [{min(get_qoa(ind) for ind in pareto_front):.4f}, "
                  f"{max(get_qoa(ind) for ind in pareto_front):.4f}]")
            print(f"  Cost range: [{min(get_cost(ind) for ind in pareto_front):.6f}, "
                  f"{max(get_cost(ind) for ind in pareto_front):.6f}]")

    return pareto_front


__all__ = [
    "run_nsga2",
    "fast_non_dominated_sort",
    "assign_crowding_distance",
    "tournament_selection",
    "deduplicate_population",
]
