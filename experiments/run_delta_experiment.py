#!/usr/bin/env python3
"""
Experiment Sweep: DP+Delta and MOQO across all levels, k, and query types.

Algorithms:
- DP+Delta: deterministic (1 run), pruning_strategy='delta_qoa', delta=0.05
- MOQO: stochastic (3 runs), no delta pruning (strict Pareto)

Output:
- results/delta_solutions.csv  — all per-solution rows
- results/delta_test_set.csv   — 6 best Pareto solutions per (level, k, query_type) joined with test_set_Opti-Q.csv
"""
import sys
import time
import gc
import csv
import random
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core.fptas import FPTAS
from src.llm_dag_optimizer.core.moqo import RandomMOQO
from old.main import clear_evaluation_caches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_QUERIES = [
    'Art', 'Geography', 'History', 'Music', 'Other',
    'Politics', 'Science and technology', 'Sports',
    'TV shows', 'Video games',
    'biology_mmlu', 'business_mmlu', 'chemistry_mmlu',
    'computer science_mmlu', 'health_mmlu', 'history_mmlu',
    'other_mmlu', 'philosophy_mmlu', 'physics_mmlu', 'psychology_mmlu',
]

# MOQO timeouts per k (seconds), from NSGA-II mean times in table_comparison.tex
MOQO_TIMEOUTS = {1: 2.0, 2: 3.4, 3: 10.2, 4: 10.1, 5: 21.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_timed(func, *args, **kwargs):
    """Run function and measure wall-clock time."""
    gc.collect()
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    return result, dt


def extract_pareto_front(solutions):
    """Extract Pareto-optimal solutions (minimize cost/latency/energy, maximize qoa)."""
    pareto = []
    for s in solutions:
        if hasattr(s, 'metrics') and s.metrics is not None:
            cost, latency, energy, qoa = s.metrics
        else:
            cost, latency, energy, qoa = s.cost, s.latency, s.energy, s.qoa
        is_dominated = False
        for other in solutions:
            if other is s:
                continue
            if hasattr(other, 'metrics') and other.metrics is not None:
                o_cost, o_lat, o_eng, o_qoa = other.metrics
            else:
                o_cost, o_lat, o_eng, o_qoa = other.cost, other.latency, other.energy, other.qoa
            if (o_cost <= cost and o_lat <= latency and o_eng <= energy and o_qoa >= qoa and
                (o_cost < cost or o_lat < latency or o_eng < energy or o_qoa > qoa)):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(s)
    return pareto


def solution_to_row(sol, algorithm, level, k, query_type, run_id, is_pareto, time_s):
    """Convert a solution Individual to a CSV row dict."""
    if hasattr(sol, 'metrics') and sol.metrics is not None:
        cost, latency, energy, qoa = sol.metrics
    else:
        cost, latency, energy, qoa = sol.cost, sol.latency, sol.energy, sol.qoa
    return {
        'algorithm': algorithm,
        'level': level,
        'k': k,
        'query_type': query_type,
        'run_id': run_id,
        'struct_id': int(sol.struct_id),
        'assignment': str(tuple(sol.assignment)),
        'estimated_cost': cost,
        'estimated_latency': latency,
        'estimated_energy': energy,
        'estimated_qoa': qoa,
        'is_pareto': is_pareto,
        'time_s': time_s,
    }


def build_best6_test_set(all_rows, test_set_path):
    """Select 6 best Pareto solutions per (algorithm, level, k, query_type), join with test questions.

    For level 4 k=1 (missing MOQO data), level 3 k=1 MOQO solutions are
    substituted with the level column set to 4.
    """
    df = pd.DataFrame(all_rows)
    pareto_df = df[df['is_pareto'] == True].copy()

    if pareto_df.empty:
        print("WARNING: No Pareto solutions found, cannot build test set.")
        return pd.DataFrame()

    # For each (algorithm, level, k, query_type), pick 6 best Pareto solutions
    best6_rows = []
    for (algo, level, k, qt), group in pareto_df.groupby(['algorithm', 'level', 'k', 'query_type']):
        deduped = group.drop_duplicates(subset=['struct_id', 'assignment'])
        deduped = deduped.sort_values(
            ['estimated_qoa', 'estimated_cost'],
            ascending=[False, True]
        )
        best6_rows.append(deduped.head(6))

    best6_df = pd.concat(best6_rows, ignore_index=True)

    # Join with test set questions (10 per query type)
    test_set = pd.read_csv(test_set_path)
    # Clean text columns: replace newlines with spaces
    for col in ['Original_Query', 'Annotated Answer']:
        if col in test_set.columns:
            test_set[col] = test_set[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)
    merged = best6_df.merge(
        test_set,
        left_on='query_type',
        right_on='Query Type',
        how='left'
    )
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Delta experiment sweep: DP+Delta and MOQO+Delta')
    parser.add_argument('--queries', nargs='+', default=ALL_QUERIES,
                        help='Query types to run (default: all 20)')
    parser.add_argument('--max-k', type=int, default=5,
                        help='Maximum k value (default: 5)')
    parser.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='Data levels to run (default: 0 1 2 3 4)')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='Delta for QoA pruning (default: 0.05)')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs for stochastic algorithms (default: 3)')
    args = parser.parse_args()

    QUERIES = args.queries
    MAX_K = args.max_k
    LEVELS = args.levels
    DELTA = args.delta
    NUM_RUNS = args.num_runs
    K_VALUES = list(range(1, MAX_K + 1))

    print("=" * 100)
    print("EXPERIMENT SWEEP: DP+Delta & MOQO")
    print("=" * 100)
    print(f"Levels     : {LEVELS}")
    print(f"k values   : {K_VALUES}")
    print(f"Queries    : {len(QUERIES)} types")
    print(f"Delta      : {DELTA} (DP only)")
    print(f"Num runs   : {NUM_RUNS} (MOQO)")
    print(f"MOQO timeouts: {MOQO_TIMEOUTS}")
    print("=" * 100)

    all_rows = []
    total_combos = len(LEVELS) * len(K_VALUES) * len(QUERIES)
    combo_idx = 0

    for level in LEVELS:
        data_path = f'data/raw/levels/level_{level}_data.csv'
        print(f"\n{'#'*100}")
        print(f"#  Level {level}: {data_path}")
        print(f"{'#'*100}")

        df_history = pd.read_csv(data_path)
        print(f"  Loaded {len(df_history)} rows")

        for k in K_VALUES:
            for query in QUERIES:
                combo_idx += 1
                print(f"\n  [{combo_idx}/{total_combos}] level={level}, k={k}, query={query}")

                # --- DP+Delta (deterministic, 1 run) ---
                clear_evaluation_caches()
                gc.collect()

                print(f"    DP+Delta ...", end=" ", flush=True)
                sols, dt = run_timed(
                    FPTAS, query, df_history, k,
                    delta=DELTA, disable_pruning=False,
                    pruning_strategy='delta_qoa', verbose=False
                )
                pareto = extract_pareto_front(sols)
                pareto_set = set(id(s) for s in pareto)
                print(f"{len(sols)} sols, {len(pareto)} Pareto, {dt:.1f}s")

                for s in sols:
                    all_rows.append(solution_to_row(
                        s, 'DP+Delta', level, k, query,
                        run_id=0, is_pareto=(id(s) in pareto_set), time_s=dt
                    ))

                # --- MOQO (stochastic, NUM_RUNS runs, no delta pruning) ---
                timeout = MOQO_TIMEOUTS.get(k, 21.0)
                for run_id in range(NUM_RUNS):
                    clear_evaluation_caches()
                    gc.collect()
                    random.seed(run_id)
                    np.random.seed(run_id)

                    print(f"    MOQO [run {run_id+1}/{NUM_RUNS}] (timeout={timeout:.1f}s) ...",
                          end=" ", flush=True)
                    sols, dt = run_timed(
                        RandomMOQO, query, df_history,
                        timeout_seconds=timeout, max_nodes=k,
                        pruning_strategy='none', delta=0.0, verbose=False
                    )
                    pareto = extract_pareto_front(sols)
                    pareto_set = set(id(s) for s in pareto)
                    print(f"{len(sols)} sols, {len(pareto)} Pareto, {dt:.1f}s")

                    for s in sols:
                        all_rows.append(solution_to_row(
                            s, 'MOQO', level, k, query,
                            run_id=run_id, is_pareto=(id(s) in pareto_set), time_s=dt
                        ))

    # --- Save results ---
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)

    solutions_file = out_dir / 'delta_solutions.csv'
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(solutions_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\nSolutions saved to: {solutions_file} ({len(df_out)} rows)")

    # --- Build best-6 test set ---
    test_set_path = Path('test_set_Opti-Q.csv')
    if test_set_path.exists():
        test_set_df = build_best6_test_set(all_rows, test_set_path)
        test_set_file = out_dir / 'delta_test_set.csv'
        test_set_df.to_csv(test_set_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Test set saved to: {test_set_file} ({len(test_set_df)} rows)")

        # Split into SimpleQA and MMLU-PRO
        if 'Category' in test_set_df.columns:
            for cat, fname in [('SimpleQA', 'simpleqa_test.csv'), ('MMLU-PRO', 'mmlu_pro_test.csv')]:
                subset = test_set_df[test_set_df['Category'] == cat]
                subset.to_csv(out_dir / fname, index=False, quoting=csv.QUOTE_NONNUMERIC)
                print(f"  {fname}: {len(subset)} rows")

        # Summary
        if not test_set_df.empty:
            print(f"\nBest-6 test set summary (rows per algorithm, level, k):")
            summary = test_set_df.groupby(['algorithm', 'level', 'k']).size().reset_index(name='rows')
            for _, row in summary.iterrows():
                print(f"  {row['algorithm']} level={int(row['level'])}, k={int(row['k'])}: {row['rows']} rows")
            print(f"  Total: {len(test_set_df)} rows")
    else:
        print(f"\nWARNING: {test_set_path} not found, skipping test set generation.")

    # --- Print summary ---
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total solution rows: {len(df_out)}")
    pareto_count = df_out[df_out['is_pareto'] == True].shape[0]
    print(f"Total Pareto solutions: {pareto_count}")
    for algo in ['DP+Delta', 'MOQO']:
        algo_df = df_out[df_out['algorithm'] == algo]
        algo_pareto = algo_df[algo_df['is_pareto'] == True]
        print(f"  {algo}: {len(algo_df)} total, {len(algo_pareto)} Pareto")


if __name__ == '__main__':
    main()
