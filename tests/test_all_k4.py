#!/usr/bin/env python3
"""
Test all three algorithms with k=4 to verify everything works.

This runs a quick test of FPTAS, MOQO, and NSGA-II with max_nodes=4
to ensure all algorithms are working correctly.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.llm_dag_optimizer.core import FPTAS, RandomMOQO, run_nsga2


def test_fptas_k4():
    """Test FPTAS with k=4."""
    print("\n" + "="*80)
    print("Testing FPTAS with k=4")
    print("="*80)

    df = pd.read_csv("data/raw/levels/level_2_data.csv")

    start_time = time.time()
    solutions = FPTAS(
        query_type="Art",
        df_history=df,
        max_nodes=4,
        epsilon=0.05,
        verbose=True,
        allow_empty_preds=True,
        extract_only_admissible=True,
        disable_pruning=False,
    )
    runtime = time.time() - start_time

    print(f"\n✓ FPTAS completed in {runtime:.2f}s")
    print(f"  Solutions: {len(solutions)}")
    if solutions:
        print(f"  QoA range: [{min(s.qoa for s in solutions):.4f}, {max(s.qoa for s in solutions):.4f}]")
        print(f"  Cost range: [{min(s.cost for s in solutions):.6f}, {max(s.cost for s in solutions):.6f}]")
        print(f"\n  Top solution:")
        best = max(solutions, key=lambda s: s.qoa)
        print(f"    QoA={best.qoa:.4f}, Cost={best.cost:.6f}, Struct={best.struct_id}, Assign={list(best.assignment)}")

    assert len(solutions) > 0, "FPTAS should return solutions"
    return solutions


def test_moqo_k4():
    """Test MOQO with k=4."""
    print("\n" + "="*80)
    print("Testing MOQO (Hill Climbing) with k=4")
    print("="*80)

    df = pd.read_csv("data/raw/levels/level_2_data.csv")

    print("Running MOQO with 30s timeout...")
    start_time = time.time()

    solutions = RandomMOQO(
        query_type="Art",
        df_history=df,
        timeout_seconds=30,
        max_nodes=4,
    )

    runtime = time.time() - start_time

    print(f"\n✓ MOQO completed in {runtime:.2f}s")
    print(f"  Solutions: {len(solutions)}")
    if solutions:
        print(f"  QoA range: [{min(s.qoa for s in solutions):.4f}, {max(s.qoa for s in solutions):.4f}]")
        print(f"  Cost range: [{min(s.cost for s in solutions):.6f}, {max(s.cost for s in solutions):.6f}]")
        print(f"\n  Top solution:")
        best = max(solutions, key=lambda s: s.qoa)
        print(f"    QoA={best.qoa:.4f}, Cost={best.cost:.6f}, Struct={best.struct_id}, Assign={list(best.assignment)}")

    assert len(solutions) > 0, "MOQO should return solutions"
    return solutions


def test_nsga2_k4():
    """Test NSGA-II with k=4."""
    print("\n" + "="*80)
    print("Testing NSGA-II with k=4")
    print("="*80)

    df = pd.read_csv("data/raw/levels/level_2_data.csv")

    start_time = time.time()
    pareto_front = run_nsga2(
        query_type="Art",
        df_history=df,
        pop_size=50,      # Smaller for quick test
        generations=20,   # Fewer for quick test
        max_nodes=4,
        verbose=True,
    )
    runtime = time.time() - start_time

    print(f"\n✓ NSGA-II completed in {runtime:.2f}s")
    print(f"  Pareto front size: {len(pareto_front)}")
    if pareto_front:
        print(f"  QoA range: [{min(ind.qoa for ind in pareto_front):.4f}, {max(ind.qoa for ind in pareto_front):.4f}]")
        print(f"  Cost range: [{min(ind.cost for ind in pareto_front):.6f}, {max(ind.cost for ind in pareto_front):.6f}]")
        print(f"\n  Top solution:")
        best = max(pareto_front, key=lambda ind: ind.qoa)
        print(f"    QoA={best.qoa:.4f}, Cost={best.cost:.6f}, Struct={best.struct_id}, Assign={list(best.assignment)}")

    assert len(pareto_front) > 0, "NSGA-II should return solutions"
    return pareto_front


def compare_algorithms(fptas_sols, moqo_sols, nsga2_sols):
    """Compare results from all three algorithms."""
    print("\n" + "="*80)
    print("COMPARISON: All Three Algorithms")
    print("="*80)

    print(f"\nSolution counts:")
    print(f"  FPTAS:   {len(fptas_sols)}")
    print(f"  MOQO:    {len(moqo_sols)}")
    print(f"  NSGA-II: {len(nsga2_sols)}")

    print(f"\nBest QoA by algorithm:")
    fptas_best_qoa = max(s.qoa for s in fptas_sols)
    moqo_best_qoa = max(s.qoa for s in moqo_sols)
    nsga2_best_qoa = max(ind.qoa for ind in nsga2_sols)

    print(f"  FPTAS:   {fptas_best_qoa:.4f}")
    print(f"  MOQO:    {moqo_best_qoa:.4f}")
    print(f"  NSGA-II: {nsga2_best_qoa:.4f}")

    print(f"\nLowest cost by algorithm:")
    fptas_min_cost = min(s.cost for s in fptas_sols)
    moqo_min_cost = min(s.cost for s in moqo_sols)
    nsga2_min_cost = min(ind.cost for ind in nsga2_sols)

    print(f"  FPTAS:   {fptas_min_cost:.6f}")
    print(f"  MOQO:    {moqo_min_cost:.6f}")
    print(f"  NSGA-II: {nsga2_min_cost:.6f}")

    print(f"\nCharacteristics:")
    print(f"  FPTAS:   Complete enumeration, deterministic")
    print(f"  MOQO:    Hill climbing, fast, time-bounded")
    print(f"  NSGA-II: Population-based, diverse solutions")


def main():
    """Main test runner."""
    print("="*80)
    print("COMPREHENSIVE TEST: All 3 Algorithms with k=4")
    print("="*80)
    print("\nThis will test FPTAS, MOQO, and NSGA-II with max_nodes=4")
    print("to verify all algorithms work correctly.\n")

    try:
        # Test FPTAS
        fptas_solutions = test_fptas_k4()

        # Test MOQO
        moqo_solutions = test_moqo_k4()

        # Test NSGA-II
        nsga2_solutions = test_nsga2_k4()

        # Compare results
        compare_algorithms(fptas_solutions, moqo_solutions, nsga2_solutions)

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\n✓ FPTAS: Working")
        print("✓ MOQO: Working")
        print("✓ NSGA-II: Working")
        print("\nAll three algorithms are functioning correctly! 🎉")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
