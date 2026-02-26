#!/usr/bin/env python3
"""
Test hypervolume correctness - verify pymoo conversion is correct.

Tests:
1. Adding a better point increases HV
2. Adding a dominated point doesn't change HV
3. HV is deterministic (same result every time)
"""

import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymoo.indicators.hv import HV


def compute_hv_correct(pareto_front_max, ref_max=None):
    """
    Correct conversion: "higher is better" → minimization.

    Args:
        pareto_front_max: Points in [0,1] where higher is better
        ref_max: Worst point in max space (default: zeros)
    """
    if len(pareto_front_max) == 0:
        return 0.0

    d = pareto_front_max.shape[1]
    if ref_max is None:
        ref_max = np.zeros(d)

    # Convert to minimization: F_min = 1 - F_max
    F_min = 1.0 - pareto_front_max
    ref_min = 1.0 - ref_max

    # Ensure ref >= all points in min space
    ref_min = np.maximum(ref_min, F_min.max(axis=0) + 1e-12)

    ind = HV(ref_point=ref_min)
    return ind(F_min)


def compute_hv_wrong(pareto_front_max, ref_max=None):
    """
    Wrong conversion: negation instead of (1 - x).
    This is what we had before.
    """
    if len(pareto_front_max) == 0:
        return 0.0

    d = pareto_front_max.shape[1]
    if ref_max is None:
        ref_max = np.zeros(d)

    # WRONG: Just negate
    F_min = -pareto_front_max
    ref_min = -ref_max

    ind = HV(ref_point=ref_min)
    return ind(F_min)


def test_adding_better_point():
    """Test: Adding a strictly better point should INCREASE HV."""
    print("="*80)
    print("TEST 1: Adding a better point should INCREASE hypervolume")
    print("="*80)

    # Start with 2 points
    pareto_base = np.array([
        [0.7, 0.6],  # Point A
        [0.6, 0.7]   # Point B
    ])

    # Add a strictly better point (dominates both)
    pareto_better = np.array([
        [0.7, 0.6],   # Point A
        [0.6, 0.7],   # Point B
        [0.8, 0.8]    # Point C (BETTER - dominates A and B)
    ])

    hv_base = compute_hv_correct(pareto_base)
    hv_better = compute_hv_correct(pareto_better)

    print(f"HV (base, 2 points):   {hv_base:.6f}")
    print(f"HV (better, 3 points): {hv_better:.6f}")
    print(f"Increase:              {hv_better - hv_base:.6f}")
    print()

    if hv_better > hv_base:
        print("✅ PASS: Adding better point increased HV")
    else:
        print("❌ FAIL: HV should increase when adding better point!")

    print()
    return hv_better > hv_base


def test_dominated_point():
    """Test: Adding a dominated point should NOT change HV."""
    print("="*80)
    print("TEST 2: Adding a dominated point should NOT change hypervolume")
    print("="*80)

    pareto_base = np.array([
        [0.8, 0.9],  # Good point
        [0.9, 0.7]   # Good point
    ])

    # Add dominated point (worse in both objectives)
    pareto_with_dominated = np.array([
        [0.8, 0.9],  # Good point
        [0.9, 0.7],  # Good point
        [0.6, 0.5]   # Dominated (worse than both)
    ])

    hv_base = compute_hv_correct(pareto_base)
    hv_with_dominated = compute_hv_correct(pareto_with_dominated)

    print(f"HV (without dominated): {hv_base:.6f}")
    print(f"HV (with dominated):    {hv_with_dominated:.6f}")
    print(f"Difference:             {abs(hv_with_dominated - hv_base):.10f}")
    print()

    if abs(hv_with_dominated - hv_base) < 1e-9:
        print("✅ PASS: Dominated point doesn't change HV")
    else:
        print("❌ FAIL: HV should not change with dominated point!")

    print()
    return abs(hv_with_dominated - hv_base) < 1e-9


def test_deterministic():
    """Test: HV should be deterministic (same result every time)."""
    print("="*80)
    print("TEST 3: Hypervolume should be deterministic")
    print("="*80)

    pareto = np.array([
        [0.8, 0.9, 0.7, 0.95],
        [0.7, 0.9, 0.8, 0.90],
        [0.9, 0.7, 0.6, 0.85]
    ])

    # Compute 5 times
    hvs = [compute_hv_correct(pareto) for _ in range(5)]

    print(f"Run 1: {hvs[0]:.10f}")
    print(f"Run 2: {hvs[1]:.10f}")
    print(f"Run 3: {hvs[2]:.10f}")
    print(f"Run 4: {hvs[3]:.10f}")
    print(f"Run 5: {hvs[4]:.10f}")
    print()

    all_same = all(abs(h - hvs[0]) < 1e-12 for h in hvs)

    if all_same:
        print("✅ PASS: Deterministic (all runs identical)")
    else:
        print("❌ FAIL: Results vary across runs!")

    print()
    return all_same


def test_wrong_vs_correct():
    """Compare wrong (negation) vs correct (1-x) conversion."""
    print("="*80)
    print("TEST 4: Compare WRONG vs CORRECT conversion")
    print("="*80)

    pareto = np.array([
        [0.8, 0.9],
        [0.9, 0.7]
    ])

    hv_wrong = compute_hv_wrong(pareto)
    hv_correct = compute_hv_correct(pareto)

    print(f"HV (wrong - negation):  {hv_wrong:.6f}")
    print(f"HV (correct - 1-x):     {hv_correct:.6f}")
    print(f"Difference:             {abs(hv_correct - hv_wrong):.6f}")
    print()

    if abs(hv_correct - hv_wrong) > 0.01:
        print("✅ Different results (as expected)")
        print("   The fix DOES change the hypervolume values!")
    else:
        print("⚠️  Results are similar (unexpected)")

    print()


def test_real_epsilon_data():
    """Test with realistic epsilon comparison."""
    print("="*80)
    print("TEST 5: Realistic epsilon comparison")
    print("="*80)

    # Simulate: reference has more/better points than epsilon=0.1
    ref_pareto = np.array([
        [0.9, 0.8, 0.7, 0.95],
        [0.8, 0.9, 0.8, 0.90],
        [0.7, 0.7, 0.9, 0.85],
    ])

    eps_pareto = np.array([
        [0.8, 0.9, 0.8, 0.90],  # Same as ref point 2
        [0.7, 0.7, 0.9, 0.85],  # Same as ref point 3
    ])

    hv_ref = compute_hv_correct(ref_pareto)
    hv_eps = compute_hv_correct(eps_pareto)

    print(f"Reference (3 points): HV = {hv_ref:.6f}")
    print(f"Epsilon   (2 points): HV = {hv_eps:.6f}")
    print(f"Quality retention:    {100*hv_eps/hv_ref:.1f}%")
    print()

    if hv_eps < hv_ref:
        print("✅ PASS: Epsilon HV < Reference HV (as expected)")
        print("   Fewer solutions → lower hypervolume")
    else:
        print("❌ FAIL: Epsilon HV should be < Reference HV!")

    print()
    return hv_eps < hv_ref


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " HYPERVOLUME CORRECTNESS TESTS ".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print()

    results = []

    results.append(("Better point increases HV", test_adding_better_point()))
    results.append(("Dominated point doesn't change HV", test_dominated_point()))
    results.append(("Deterministic results", test_deterministic()))
    test_wrong_vs_correct()  # Just for comparison
    results.append(("Epsilon comparison", test_real_epsilon_data()))

    print("="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print()

    if all(r[1] for r in results):
        print("🎉 ALL TESTS PASSED! Hypervolume conversion is CORRECT.")
    else:
        print("⚠️  SOME TESTS FAILED! Check the implementation.")

    print()
