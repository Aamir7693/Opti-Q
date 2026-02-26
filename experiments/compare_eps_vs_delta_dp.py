#!/usr/bin/env python3
"""
Epsilon vs Delta pruning experiment for DP (FPTAS).

Sweeps epsilon and delta values across k=1..5, all queries.
- Baseline: DP no pruning (HV=100% reference)
- Epsilon: pruning_strategy='epsilon', sweep epsilon values
- Delta:   pruning_strategy='delta_qoa', sweep delta values

Cache cleared before EVERY FPTAS call for full isolation.
Level 2 data, fuzzy matching OFF.
"""
import sys
import time
import gc
import os
import argparse
from pathlib import Path

import psutil
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core.fptas import FPTAS
from old.main import clear_evaluation_caches

try:
    import pygmo as pg
    from experiments.hypervolume_pygmo import normalize_objectives, compute_hypervolume_pygmo, get_pareto_front
    USE_PYGMO = True
except ImportError:
    USE_PYGMO = False


# ---------------------------------------------------------------------------
# Helpers (same as compare_dp_hc_nsga.py)
# ---------------------------------------------------------------------------

def extract_pareto_front_numpy(points):
    if len(points) == 0:
        return points
    is_pareto = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if is_pareto[i]:
            for j, other in enumerate(points):
                if i != j and is_pareto[j]:
                    if np.all(other >= point) and np.any(other > point):
                        is_pareto[i] = False
                        break
    return points[is_pareto]


def compute_hypervolume(solutions, global_mins=None, global_maxs=None, return_bounds=False):
    if not solutions:
        return (0.0, None, None) if return_bounds else 0.0
    if USE_PYGMO:
        normalized, mins, maxs = normalize_objectives(solutions, global_mins, global_maxs)
        pareto = get_pareto_front(normalized)
        hv_value = compute_hypervolume_pygmo(pareto)
    else:
        raw = np.array([s.metrics for s in solutions])
        mins = global_mins if global_mins is not None else np.min(raw, axis=0)
        maxs = global_maxs if global_maxs is not None else np.max(raw, axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = (raw - mins) / ranges
        normalized_max = normalized.copy()
        normalized_max[:, 0] = 1.0 - normalized_max[:, 0]
        normalized_max[:, 1] = 1.0 - normalized_max[:, 1]
        normalized_max[:, 2] = 1.0 - normalized_max[:, 2]
        pareto_max = extract_pareto_front_numpy(normalized_max)
        F_min = 1.0 - pareto_max
        ref_point = np.maximum(np.ones(4), F_min.max(axis=0) + 1e-12)
        try:
            from pymoo.indicators.hv import HV
            hv = HV(ref_point=ref_point)
            hv_value = hv(F_min)
        except Exception as e:
            print(f"Warning: HV computation failed: {e}")
            hv_value = 0.0
    return (hv_value, mins, maxs) if return_bounds else hv_value


def extract_pareto_front(solutions):
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


def get_best_qoa(solutions):
    if not solutions:
        return 0.0
    qoas = []
    for s in solutions:
        if hasattr(s, 'metrics') and s.metrics is not None:
            qoas.append(s.metrics[3])
        else:
            qoas.append(s.qoa)
    return max(qoas)


def run_with_psutil(func, *args, **kwargs):
    import threading
    gc.collect()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    peak_rss = rss_before
    stop_event = threading.Event()

    def _poll_rss():
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                current = process.memory_info().rss
                if current > peak_rss:
                    peak_rss = current
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop_event.wait(0.05)

    monitor = threading.Thread(target=_poll_rss, daemon=True)
    monitor.start()
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    stop_event.set()
    monitor.join(timeout=1.0)
    try:
        final_rss = process.memory_info().rss
        if final_rss > peak_rss:
            peak_rss = final_rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    mem_mb = max(0, peak_rss - rss_before) / 1024 / 1024
    return result, dt, mem_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Epsilon vs Delta pruning for DP (FPTAS)')
    parser.add_argument('--queries', nargs='+',
                        default=['Art', 'Geography', 'History', 'Music', 'Other',
                                 'Politics', 'Science and technology', 'Sports',
                                 'TV shows', 'Video games'])
    parser.add_argument('--max-k', type=int, default=5)
    parser.add_argument('--epsilon-values', nargs='+', type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument('--delta-values', nargs='+', type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    args = parser.parse_args()

    QUERIES = args.queries
    MAX_K = args.max_k
    K_VALUES = list(range(1, MAX_K + 1))
    EPS_VALUES = args.epsilon_values
    DELTA_VALUES = args.delta_values

    df_history = pd.read_csv('data/raw/levels/level_2_data.csv')

    print("=" * 120)
    print("EPSILON vs DELTA PRUNING: DP (FPTAS)")
    print("=" * 120)
    print(f"Data       : level_2_data.csv ({len(df_history)} rows), fuzzy matching OFF")
    print(f"Queries    : {QUERIES}")
    print(f"k values   : {K_VALUES}")
    print(f"Epsilon    : {EPS_VALUES}")
    print(f"Delta      : {DELTA_VALUES}")
    print(f"HV Method  : {'pygmo' if USE_PYGMO else 'pymoo'}")
    print(f"Cache      : cleared before EVERY FPTAS call")
    print("=" * 120)

    rows = []  # CSV output rows

    for k in K_VALUES:
        print(f"\n\n{'#'*120}")
        print(f"#  k = {k}")
        print(f"{'#'*120}")

        # ==================================================================
        # Step 1: DP no pruning baseline (per query)
        # ==================================================================
        baselines = {}  # {query: {hv, g_mins, g_maxs}}
        baseline_data = {}  # {query: (sols, pareto, dt, mem, hv)}

        print(f"\n--- DP No Pruning (baseline, k={k}) ---")
        for q in QUERIES:
            clear_evaluation_caches()
            gc.collect()

            print(f"  {q} ...", end=" ", flush=True)
            sols, dt, mem = run_with_psutil(
                FPTAS, q, df_history, k,
                epsilon=0.0, disable_pruning=True,
                pruning_strategy=None, verbose=False
            )
            _, g_mins, g_maxs = compute_hypervolume(sols, return_bounds=True)
            pareto = extract_pareto_front(sols)
            hv = compute_hypervolume(pareto, g_mins, g_maxs)
            baselines[q] = {'hv': hv, 'g_mins': g_mins, 'g_maxs': g_maxs}
            baseline_data[q] = (sols, pareto, dt, mem, hv)

            b_hv = baselines[q]['hv']
            hv_ret = (hv / b_hv * 100) if b_hv > 0 else 0.0
            qoa = get_best_qoa(sols)
            print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, "
                  f"HV%={hv_ret:.1f}, {dt:.1f}s, {mem:.1f}MB")

            rows.append({
                'k': k, 'query': q, 'strategy': 'none', 'param': 0.0,
                'solutions': len(sols), 'pareto': len(pareto),
                'hv': hv, 'hv_retention_pct': hv_ret,
                'best_qoa': qoa, 'time_s': dt, 'memory_mb': mem,
            })

        # ==================================================================
        # Step 2: Sweep epsilon values
        # ==================================================================
        for eps in EPS_VALUES:
            print(f"\n--- DP + Epsilon (eps={eps}, k={k}) ---")
            for q in QUERIES:
                clear_evaluation_caches()
                gc.collect()

                print(f"  {q} ...", end=" ", flush=True)
                sols, dt, mem = run_with_psutil(
                    FPTAS, q, df_history, k,
                    epsilon=eps, disable_pruning=False,
                    pruning_strategy='epsilon', verbose=False
                )
                pareto = extract_pareto_front(sols)
                hv = compute_hypervolume(pareto, baselines[q]['g_mins'], baselines[q]['g_maxs'])
                b_hv = baselines[q]['hv']
                hv_ret = (hv / b_hv * 100) if b_hv > 0 else 0.0
                qoa = get_best_qoa(sols)
                print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, "
                      f"HV%={hv_ret:.1f}, {dt:.1f}s, {mem:.1f}MB")

                rows.append({
                    'k': k, 'query': q, 'strategy': 'epsilon', 'param': eps,
                    'solutions': len(sols), 'pareto': len(pareto),
                    'hv': hv, 'hv_retention_pct': hv_ret,
                    'best_qoa': qoa, 'time_s': dt, 'memory_mb': mem,
                })

        # ==================================================================
        # Step 3: Sweep delta values
        # ==================================================================
        for delta in DELTA_VALUES:
            print(f"\n--- DP + Delta (delta={delta}, k={k}) ---")
            for q in QUERIES:
                clear_evaluation_caches()
                gc.collect()

                print(f"  {q} ...", end=" ", flush=True)
                sols, dt, mem = run_with_psutil(
                    FPTAS, q, df_history, k,
                    delta=delta, disable_pruning=False,
                    pruning_strategy='delta_qoa', verbose=False
                )
                pareto = extract_pareto_front(sols)
                hv = compute_hypervolume(pareto, baselines[q]['g_mins'], baselines[q]['g_maxs'])
                b_hv = baselines[q]['hv']
                hv_ret = (hv / b_hv * 100) if b_hv > 0 else 0.0
                qoa = get_best_qoa(sols)
                print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, "
                      f"HV%={hv_ret:.1f}, {dt:.1f}s, {mem:.1f}MB")

                rows.append({
                    'k': k, 'query': q, 'strategy': 'delta_qoa', 'param': delta,
                    'solutions': len(sols), 'pareto': len(pareto),
                    'hv': hv, 'hv_retention_pct': hv_ret,
                    'best_qoa': qoa, 'time_s': dt, 'memory_mb': mem,
                })

        # ==================================================================
        # Print per-k summary
        # ==================================================================
        print(f"\n{'='*120}")
        print(f"SUMMARY k={k} (avg across {len(QUERIES)} queries)")
        print(f"{'='*120}\n")
        print(f"{'Strategy':<14} {'Param':<8} {'Time(s)':<10} {'Mem(MB)':<10} "
              f"{'Solutions':<12} {'Pareto':<10} {'HV%':<10} {'BestQoA':<10}")
        print("-" * 94)

        # Baseline row
        k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'none']
        if k_rows:
            avg = lambda f: np.mean([r[f] for r in k_rows])
            print(f"{'none':<14} {'--':<8} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                  f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                  f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")

        # Epsilon rows
        for eps in EPS_VALUES:
            k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'epsilon' and r['param'] == eps]
            if k_rows:
                avg = lambda f: np.mean([r[f] for r in k_rows])
                print(f"{'epsilon':<14} {eps:<8.2f} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                      f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                      f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")

        # Delta rows
        for delta in DELTA_VALUES:
            k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'delta_qoa' and r['param'] == delta]
            if k_rows:
                avg = lambda f: np.mean([r[f] for r in k_rows])
                print(f"{'delta_qoa':<14} {delta:<8.2f} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                      f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                      f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")

    # ======================================================================
    # Grand summary
    # ======================================================================
    print(f"\n\n{'#'*120}")
    print(f"#  GRAND SUMMARY (avg across {len(QUERIES)} queries)")
    print(f"{'#'*120}\n")
    print(f"{'k':<4} {'Strategy':<14} {'Param':<8} {'Time(s)':<10} {'Mem(MB)':<10} "
          f"{'Solutions':<12} {'Pareto':<10} {'HV%':<10} {'BestQoA':<10}")
    print("-" * 98)

    for k in K_VALUES:
        # Baseline
        k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'none']
        if k_rows:
            avg = lambda f: np.mean([r[f] for r in k_rows])
            print(f"{k:<4} {'none':<14} {'--':<8} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                  f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                  f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")

        for eps in EPS_VALUES:
            k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'epsilon' and r['param'] == eps]
            if k_rows:
                avg = lambda f: np.mean([r[f] for r in k_rows])
                print(f"{k:<4} {'epsilon':<14} {eps:<8.2f} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                      f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                      f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")

        for delta in DELTA_VALUES:
            k_rows = [r for r in rows if r['k'] == k and r['strategy'] == 'delta_qoa' and r['param'] == delta]
            if k_rows:
                avg = lambda f: np.mean([r[f] for r in k_rows])
                print(f"{k:<4} {'delta_qoa':<14} {delta:<8.2f} {avg('time_s'):<10.1f} {avg('memory_mb'):<10.1f} "
                      f"{avg('solutions'):<12.0f} {avg('pareto'):<10.0f} "
                      f"{avg('hv_retention_pct'):<10.1f} {avg('best_qoa'):<10.4f}")
        print()

    # ======================================================================
    # Save CSV
    # ======================================================================
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'eps_vs_delta_dp.csv'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
