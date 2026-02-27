#!/usr/bin/env python3
"""
Standalone MOQO (RandomMOQO) experiment script.

Runs MOQO (no pruning) always. If --delta is provided, also runs MOQO+Delta.

Also creates a test-set file with 6 best solutions per query type matched
to questions from test_set_Opti-Q.csv.

Usage:
    python experiments/run_moqo.py --queries Art --max-k 5 --timeout 20 --num-runs 3 --delta 0.05 --level 2
"""
import sys
import time
import gc
import random
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core.moqo import RandomMOQO
from old.main import clear_evaluation_caches
from experiments.compare_dp_hc_nsga import extract_pareto_front

ALL_QUERIES = [
    'Art', 'Geography', 'History', 'Music', 'Other',
    'Politics', 'Science and technology', 'Sports',
    'TV shows', 'Video games',
]


def get_metrics(s):
    if hasattr(s, 'metrics') and s.metrics is not None:
        return s.metrics
    return (s.cost, s.latency, s.energy, s.qoa)


def build_best6_test_set(df_solutions, out_file):
    """Pick 6 best Pareto solutions per query_type and join with test_set_Opti-Q.csv."""
    test_set_path = Path('test_set_Opti-Q.csv')
    if not test_set_path.exists():
        print(f"  WARNING: {test_set_path} not found, skipping test-set file.")
        return

    df_test = pd.read_csv(test_set_path)

    # Pick best 6 Pareto solutions per query_type (highest estimated_qoa, deduped)
    pareto = df_solutions[df_solutions['is_pareto']].copy()
    pareto = pareto.drop_duplicates(subset=['query_type', 'struct_id', 'assignment'])
    best6 = (
        pareto
        .sort_values('estimated_qoa', ascending=False)
        .groupby('query_type')
        .head(6)
        .reset_index(drop=True)
    )
    best6['solution_rank'] = best6.groupby('query_type').cumcount() + 1

    # Cross-join: each question gets its query_type's 6 best solutions
    merged = df_test.merge(best6, left_on='Query Type', right_on='query_type', how='inner')
    merged = merged.drop(columns=['query_type'])
    merged = merged.sort_values(['Query Type', 'Original_Query', 'solution_rank'])

    merged.to_csv(out_file, index=False)
    print(f"Saved {len(merged)} rows to {out_file}")


def run_moqo_variant(algo_name, query_type, df_history, k, level, timeout,
                     run_id, delta=None):
    """Run one MOQO variant for one (k, query, run) and return row dicts."""
    clear_evaluation_caches()
    gc.collect()
    random.seed(run_id)
    np.random.seed(run_id)

    if delta is not None:
        pruning_strategy = 'delta_qoa'
    else:
        pruning_strategy = 'none'

    print(f"k={k} | {query_type} | {algo_name} | run {run_id+1} ...", end=" ", flush=True)

    t0 = time.time()
    sols = RandomMOQO(
        query_type, df_history,
        timeout_seconds=timeout,
        max_nodes=k,
        pruning_strategy=pruning_strategy,
        delta=delta if delta is not None else 0.01,
        verbose=False,
    )
    dt = time.time() - t0

    pareto = extract_pareto_front(sols)
    pareto_set = set(id(s) for s in pareto)

    print(f"{len(sols)} sols, {len(pareto)} Pareto, {dt:.1f}s")

    rows = []
    for s in sols:
        cost, latency, energy, qoa = get_metrics(s)
        rows.append({
            'algorithm': algo_name,
            'level': level,
            'k': k,
            'query_type': query_type,
            'run_id': run_id,
            'struct_id': int(s.struct_id),
            'assignment': str(tuple(s.assignment)),
            'estimated_cost': cost,
            'estimated_latency': latency,
            'estimated_energy': energy,
            'estimated_qoa': qoa,
            'is_pareto': id(s) in pareto_set,
            'time_s': dt,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Run MOQO experiment')
    parser.add_argument('--queries', nargs='+', default=ALL_QUERIES)
    parser.add_argument('--max-k', type=int, default=5)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--num-runs', type=int, default=3)
    parser.add_argument('--timeout', type=float, required=True,
                        help='Timeout in seconds per (k, query, run)')
    parser.add_argument('--delta', type=float, default=None,
                        help='If provided, also run MOQO+Delta with this value')
    args = parser.parse_args()

    df_history = pd.read_csv(f'data/raw/levels/level_{args.level}_data.csv')

    variants = ['MOQO']
    if args.delta is not None:
        variants.append('MOQO+Delta')

    print("=" * 80)
    print("MOQO (RandomMOQO) Standalone Experiment")
    print("=" * 80)
    print(f"Level      : {args.level} ({len(df_history)} rows)")
    print(f"Queries    : {args.queries}")
    print(f"k values   : 1..{args.max_k}")
    print(f"Timeout    : {args.timeout}s")
    print(f"Variants   : {variants}")
    print(f"Num runs   : {args.num_runs}")
    if args.delta is not None:
        print(f"Delta      : {args.delta}")
    print("=" * 80)

    rows = []

    for k in range(1, args.max_k + 1):
        for q in args.queries:
            for run_id in range(args.num_runs):
                # Always run MOQO (no pruning)
                rows.extend(run_moqo_variant(
                    'MOQO', q, df_history, k, args.level,
                    args.timeout, run_id,
                ))

                # Optionally run MOQO+Delta
                if args.delta is not None:
                    rows.extend(run_moqo_variant(
                        'MOQO+Delta', q, df_history, k, args.level,
                        args.timeout, run_id, delta=args.delta,
                    ))

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'moqo_solutions.csv'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {len(df_out)} rows to {out_file}")

    # Build test-set file with 6 best solutions per query type
    build_best6_test_set(df_out, out_dir / 'moqo_test_set.csv')

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    summary = df_out.groupby(['algorithm', 'k', 'query_type', 'run_id']).agg(
        solutions=('struct_id', 'count'),
        pareto=('is_pareto', 'sum'),
        time_s=('time_s', 'first'),
        best_qoa=('estimated_qoa', 'max'),
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
