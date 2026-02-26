#!/usr/bin/env python3
"""
Production Hypervolume using pygmo (WFG Algorithm - Exact)

Installation:
    pip install pygmo

pygmo uses the WFG algorithm which is:
✅ Exact (not approximation like Monte Carlo)
✅ Fast (C++ implementation)
✅ Industry standard
✅ Handles any dimension (2D, 4D, 10D, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    import pygmo as pg
    PYGMO_AVAILABLE = True
except ImportError:
    PYGMO_AVAILABLE = False
    print("⚠️  pygmo not installed. Install with: pip install pygmo")


def normalize_objectives(
    solutions: List,
    global_mins: Optional[np.ndarray] = None,
    global_maxs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize objectives to [0,1] with 'higher is better'.

    Args:
        solutions: List of solution objects
        global_mins: Fixed minimum values (for epsilon comparison)
        global_maxs: Fixed maximum values (for epsilon comparison)

    Returns:
        (normalized, mins, maxs)
    """
    # Extract metrics
    metrics = []
    for s in solutions:
        if hasattr(s, 'metrics'):
            cost, latency, energy, qoa = s.metrics
        else:
            cost = s.cost
            latency = s.latency
            energy = s.energy
            qoa = s.qoa
        metrics.append([cost, latency, energy, qoa])

    metrics = np.array(metrics)

    # Use global bounds if provided
    if global_mins is None:
        mins = metrics.min(axis=0)
    else:
        mins = global_mins

    if global_maxs is None:
        maxs = metrics.max(axis=0)
    else:
        maxs = global_maxs

    # Normalize to [0,1]
    ranges = maxs - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    normalized = (metrics - mins) / ranges

    # Invert minimize objectives
    normalized[:, 0] = 1.0 - normalized[:, 0]  # Cost
    normalized[:, 1] = 1.0 - normalized[:, 1]  # Latency
    normalized[:, 2] = 1.0 - normalized[:, 2]  # Energy
    # QoA already maximize

    return normalized, mins, maxs


def get_pareto_front(points: np.ndarray) -> np.ndarray:
    """Extract Pareto front (non-dominated points)."""
    is_pareto = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if is_pareto[i]:
            for j, other in enumerate(points):
                if i != j and is_pareto[j]:
                    if np.all(other >= point) and np.any(other > point):
                        is_pareto[i] = False
                        break

    return points[is_pareto]


def compute_hypervolume_pygmo(
    pareto_front_max: np.ndarray,
    reference_point_max: Optional[np.ndarray] = None
) -> float:
    """
    Compute exact hypervolume using pygmo's WFG algorithm (CORRECT conversion).

    pygmo expects MINIMIZATION where reference is the WORST point.

    Our normalization: "higher is better" in [0,1]
    - Best corner: [1,1,1,1]
    - Worst corner: [0,0,0,0]

    pygmo minimization: "lower is better"
    - Convert: F_min = 1 - F_max
    - Worst corner becomes: [1,1,1,1] in min space
    - Best corner becomes: [0,0,0,0] in min space

    Args:
        pareto_front_max: n × d array, "higher is better" in [0,1]
        reference_point_max: Worst point in max space (default: zeros)

    Returns:
        Exact hypervolume

    Example:
        >>> pareto = np.array([[0.8, 0.9], [0.9, 0.7]])
        >>> hv = compute_hypervolume_pygmo(pareto)
        >>> print(f"HV: {hv:.6f}")
    """
    if not PYGMO_AVAILABLE:
        raise ImportError("pygmo not installed. Install with: pip install pygmo")

    if len(pareto_front_max) == 0:
        return 0.0

    d = pareto_front_max.shape[1]

    # Default reference in MAX space is worst corner (0s)
    if reference_point_max is None:
        reference_point_max = np.zeros(d)

    # Convert to MINIMIZATION: F_min = 1 - F_max
    F_min = 1.0 - pareto_front_max
    ref_min = 1.0 - reference_point_max

    # Safety: Ensure ref is >= all points in minimization
    # (pygmo expects reference to be dominated by all points)
    ref_min = np.maximum(ref_min, F_min.max(axis=0) + 1e-12)

    # Create hypervolume object and compute
    hv_obj = pg.hypervolume(F_min.tolist())
    return hv_obj.compute(ref_min.tolist())


def compute_hypervolume_4d_pygmo(
    solutions: List,
    global_mins: Optional[np.ndarray] = None,
    global_maxs: Optional[np.ndarray] = None
) -> Tuple[float, int]:
    """
    High-level API: Compute 4D hypervolume from solutions.

    Args:
        solutions: List of solution objects
        global_mins: Fixed bounds for epsilon comparison
        global_maxs: Fixed bounds for epsilon comparison

    Returns:
        (hypervolume, n_pareto_points)
    """
    # Normalize
    normalized, mins, maxs = normalize_objectives(solutions, global_mins, global_maxs)

    # Get Pareto front
    pareto = get_pareto_front(normalized)

    # Compute HV
    hv = compute_hypervolume_pygmo(pareto)

    return hv, len(pareto)


def compare_epsilon_with_pygmo(
    reference_solutions: List,
    epsilon_solutions_dict: dict
) -> dict:
    """
    Compare epsilon values using exact pygmo hypervolume.

    Args:
        reference_solutions: Ground truth (ε≈0)
        epsilon_solutions_dict: {epsilon: solutions}

    Returns:
        Results dictionary with HV comparisons
    """
    # Get reference HV and global bounds
    ref_norm, global_mins, global_maxs = normalize_objectives(reference_solutions)
    ref_pareto = get_pareto_front(ref_norm)
    ref_hv = compute_hypervolume_pygmo(ref_pareto)

    print(f"Reference: {len(ref_pareto)} Pareto points, HV = {ref_hv:.6f}")
    print()

    # Compare each epsilon
    results = {}
    for eps in sorted(epsilon_solutions_dict.keys()):
        solutions = epsilon_solutions_dict[eps]

        # Normalize with same bounds
        normalized, _, _ = normalize_objectives(solutions, global_mins, global_maxs)
        pareto = get_pareto_front(normalized)
        hv = compute_hypervolume_pygmo(pareto)

        results[eps] = {
            'hv': hv,
            'hv_percent': 100 * hv / ref_hv if ref_hv > 0 else 0,
            'n_pareto': len(pareto)
        }

        print(f"ε={eps:6.4f}: HV={hv:.6f} ({results[eps]['hv_percent']:5.1f}%), "
              f"{len(pareto):3d} Pareto points")

    return {
        'reference_hv': ref_hv,
        'global_mins': global_mins,
        'global_maxs': global_maxs,
        'results': results
    }


if __name__ == "__main__":
    if not PYGMO_AVAILABLE:
        print("="*80)
        print("pygmo is not installed!")
        print("="*80)
        print()
        print("Install with:")
        print("  pip install pygmo")
        print()
        print("Or using conda:")
        print("  conda install -c conda-forge pygmo")
        print()
        exit(1)

    print("="*80)
    print("HYPERVOLUME WITH PYGMO (EXACT WFG ALGORITHM)")
    print("="*80)
    print()

    # Example
    print("Example: 4D Hypervolume")
    print("-" * 80)

    pareto_front = np.array([
        [0.9, 0.8, 0.7, 0.95],
        [0.7, 0.9, 0.8, 0.90],
        [0.5, 0.7, 0.6, 0.85]
    ])

    print(f"Pareto front: {len(pareto_front)} points")
    hv = compute_hypervolume_pygmo(pareto_front)
    print(f"4D Hypervolume (exact): {hv:.6f}")
    print()

    print("="*80)
    print("✅ pygmo is FAST, EXACT, and PRODUCTION-READY!")
    print("="*80)
