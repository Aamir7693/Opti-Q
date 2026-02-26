#!/usr/bin/env python3
"""
Standalone DP (FPTAS) experiment script.

Runs DP (no pruning) always. If --delta is provided, also runs DP+Delta.
Deterministic — no --num-runs.

Also creates a test-set file with 6 best solutions per query type matched
to questions from test_set_Opti-Q.csv.

Usage:
    python experiments/run_dp.py --queries Art --max-k 5 --delta 0.05 --level 2
"""
import sys
import time
import gc
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core.fptas import FPTAS
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


def run_dp_variant(algo_name, query_type, df_history, k, level, delta=None):
    """Run one DP variant and return list of row dicts."""
    clear_evaluation_caches()
    gc.collect()

    if delta is not None:
        disable_pruning = False
        pruning_strategy = 'delta_qoa'
    else:
        disable_pruning = True
        pruning_strategy = None

    print(f"k={k} | {query_type} | {algo_name} ...", end=" ", flush=True)

    t0 = time.time()
    sols = FPTAS(
        query_type, df_history, k,
        epsilon=0.0,
        delta=delta if delta is not None else 0.05,
        disable_pruning=disable_pruning,
        pruning_strategy=pruning_strategy,
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
            'run_id': 0,
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
    parser = argparse.ArgumentParser(description='Run DP (FPTAS) experiment')
    parser.add_argument('--queries', nargs='+', default=ALL_QUERIES)
    parser.add_argument('--max-k', type=int, default=5)
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--delta', type=float, default=None,
                        help='If provided, also run DP+Delta with this value')
    parser.add_argument('--only-delta', action='store_true',
                        help='Run only DP+Delta (requires --delta)')
    args = parser.parse_args()

    if args.only_delta and args.delta is None:
        parser.error('--only-delta requires --delta')

    df_history = pd.read_csv(f'data/raw/levels/level_{args.level}_data.csv')

    variants = []
    if not args.only_delta:
        variants.append('DP')
    if args.delta is not None:
        variants.append('DP+Delta')

    print("=" * 80)
    print("DP (FPTAS) Standalone Experiment")
    print("=" * 80)
    print(f"Level      : {args.level} ({len(df_history)} rows)")
    print(f"Queries    : {args.queries}")
    print(f"k values   : 1..{args.max_k}")
    print(f"Variants   : {variants}")
    if args.delta is not None:
        print(f"Delta      : {args.delta}")
    print("=" * 80)

    rows = []

    for k in range(1, args.max_k + 1):
        for q in args.queries:
            # Run DP (no pruning) unless --only-delta
            if not args.only_delta:
                rows.extend(run_dp_variant('DP', q, df_history, k, args.level))

            # Run DP+Delta if --delta provided
            if args.delta is not None:
                rows.extend(run_dp_variant('DP+Delta', q, df_history, k, args.level, delta=args.delta))

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'dp_solutions.csv'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {len(df_out)} rows to {out_file}")

    # Build test-set file with 6 best solutions per query type
    build_best6_test_set(df_out, out_dir / 'dp_test_set.csv')

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    summary = df_out.groupby(['algorithm', 'k', 'query_type']).agg(
        solutions=('struct_id', 'count'),
        pareto=('is_pareto', 'sum'),
        time_s=('time_s', 'first'),
        best_qoa=('estimated_qoa', 'max'),
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
