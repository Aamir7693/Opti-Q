#!/usr/bin/env python3
"""
Compare DP vs MOQO (HC) vs NSGA-II across k=1..5.
- NSGA-II
- DP no pruning (baseline, provides HV reference bounds)
- DP + delta pruning (delta=0.05)
- MOQO (HC) no pruning
- MOQO (HC) + delta pruning (delta=0.05)

MOQO timeout = max NSGA-II mean time across queries for that k.
HV normalized using DP baseline bounds per query per k.
All solutions are from k=1..max_k (accumulated Pareto).

Stochastic algorithms (NSGA-II, MOQO, MOQO+Delta) are run --num-runs times
and report mean +/- std. DP and DP+Delta are deterministic (run once).
"""
import sys
import time
import gc
import os
import random
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core.fptas import FPTAS
from src.llm_dag_optimizer.core.nsga2 import run_nsga2
from src.llm_dag_optimizer.core.moqo import RandomMOQO
from old.main import clear_evaluation_caches

try:
    import pygmo as pg
    from experiments.hypervolume_pygmo import normalize_objectives, compute_hypervolume_pygmo, get_pareto_front
    USE_PYGMO = True
except ImportError:
    USE_PYGMO = False

from pymoo.indicators.igd import IGD


# ---------------------------------------------------------------------------
# Helpers
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


def compute_igd(solutions, ref_pareto_min, global_mins, global_maxs):
    """Compute IGD of solutions against DP reference Pareto front.

    Both inputs are normalized to [0,1] minimization space using DP bounds.
    ref_pareto_min: DP's Pareto front in minimization space (precomputed).
    Returns IGD value (lower is better).
    """
    if not solutions or ref_pareto_min is None or len(ref_pareto_min) == 0:
        return float('inf')
    # Normalize solutions using DP bounds
    raw = []
    for s in solutions:
        if hasattr(s, 'metrics') and s.metrics is not None:
            raw.append(list(s.metrics))
        else:
            raw.append([s.cost, s.latency, s.energy, s.qoa])
    raw = np.array(raw)
    ranges = global_maxs - global_mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    normalized = (raw - global_mins) / ranges
    normalized[:, 0] = 1.0 - normalized[:, 0]  # Cost
    normalized[:, 1] = 1.0 - normalized[:, 1]  # Latency
    normalized[:, 2] = 1.0 - normalized[:, 2]  # Energy
    # Now "higher is better" in [0,1]. Convert to minimization for IGD.
    F_min = 1.0 - normalized
    # Extract Pareto front of the approximation
    pareto_mask = np.ones(len(F_min), dtype=bool)
    for i in range(len(F_min)):
        if pareto_mask[i]:
            for j in range(len(F_min)):
                if i != j and pareto_mask[j]:
                    if np.all(F_min[j] <= F_min[i]) and np.any(F_min[j] < F_min[i]):
                        pareto_mask[i] = False
                        break
    approx_pareto_min = F_min[pareto_mask]
    if len(approx_pareto_min) == 0:
        return float('inf')
    igd = IGD(ref_pareto_min)
    return igd(approx_pareto_min)


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


def run_timed(func, *args, **kwargs):
    """Run function and measure wall-clock time only."""
    gc.collect()

    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0

    return result, dt


def fmt_mean_std(mean, std, precision=1):
    """Format mean +/- std string, or plain value if deterministic (std is None)."""
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', nargs='+',
                        default=['Art', 'Geography', 'History', 'Music', 'Other',
                                 'Politics', 'Science and technology', 'Sports',
                                 'TV shows', 'Video games'])
    parser.add_argument('--max-k', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--nsga-pop', type=int, default=200)
    parser.add_argument('--nsga-gen', type=int, default=200)
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs for stochastic algorithms (NSGA-II, MOQO)')
    parser.add_argument('--reuse-dp', action='store_true',
                        help='Reuse cached DP results from previous run')
    args = parser.parse_args()

    QUERIES = args.queries
    MAX_K = args.max_k
    DELTA = args.delta
    NUM_RUNS = args.num_runs
    K_VALUES = list(range(1, MAX_K + 1))

    STOCHASTIC_ALGOS = {'NSGA-II', 'MOQO', 'MOQO+Delta'}

    df_history = pd.read_csv('data/raw/levels/level_2_data.csv')

    print("=" * 120)
    print("ALGORITHM COMPARISON: DP vs MOQO (HC) vs NSGA-II")
    print("=" * 120)
    print(f"Data       : level_2_data.csv ({len(df_history)} rows), fuzzy matching OFF")
    print(f"Queries    : {QUERIES}")
    print(f"k values   : {K_VALUES}")
    print(f"Delta      : {DELTA}")
    print(f"NSGA-II    : pop={args.nsga_pop}, gen={args.nsga_gen}")
    print(f"Num runs   : {NUM_RUNS} (stochastic algos)")
    print(f"Memory     : disabled")
    print(f"HV Method  : {'pygmo' if USE_PYGMO else 'pymoo'}")
    print("=" * 120)

    # Storage: all_results[k][query][algo] = {
    #   'time_mean', 'time_std',
    #   'sols_mean', 'sols_std', 'pareto_mean', 'pareto_std',
    #   'hv_mean', 'hv_std', 'hv_ret_mean', 'hv_ret_std',
    #   'qoa_mean', 'qoa_std', 'runs': [{sols, pareto, dt, hv, hv_ret, qoa}, ...]
    # }
    all_results = {}

    for k in K_VALUES:
        print(f"\n\n{'#'*120}")
        print(f"#  k = {k}")
        print(f"{'#'*120}")

        all_results[k] = {}

        # Clear evaluation caches so each k starts fresh
        clear_evaluation_caches()
        gc.collect()

        # ------------------------------------------------------------------
        # Step 1: Run NSGA-II on all queries (NUM_RUNS times each)
        # ------------------------------------------------------------------
        nsga_run_data = {}  # {query: [list of (sols, dt) per run]}

        print(f"\n--- NSGA-II (k={k}, {NUM_RUNS} runs) ---")
        for q in QUERIES:
            nsga_run_data[q] = []
            for run_id in range(NUM_RUNS):
                # Clear caches and re-seed RNG so each run is independent
                clear_evaluation_caches()
                gc.collect()
                random.seed(run_id)
                np.random.seed(run_id)
                print(f"  {q} [run {run_id+1}/{NUM_RUNS}] ...", end=" ", flush=True)
                sols, dt = run_timed(
                    run_nsga2, query_type=q, df_history=df_history,
                    max_nodes=k, pop_size=args.nsga_pop, generations=args.nsga_gen,
                    verbose=False
                )
                nsga_run_data[q].append((sols, dt))
                print(f"{len(sols)} sols, {dt:.1f}s")

        # MOQO timeout = per-query mean NSGA-II time (fair: each query gets same budget)
        nsga_mean_times = {q: np.mean([r[1] for r in nsga_run_data[q]]) for q in QUERIES}
        moqo_timeout_per_query = {q: nsga_mean_times[q] for q in QUERIES}
        moqo_timeout_max = max(nsga_mean_times.values())
        moqo_timeout_median = float(np.median(list(nsga_mean_times.values())))
        print(f"  NSGA-II mean times per query:")
        for q in QUERIES:
            print(f"    {q}: {nsga_mean_times[q]:.1f}s")
        print(f"  Median: {moqo_timeout_median:.1f}s, Max: {moqo_timeout_max:.1f}s")
        print(f"  MOQO timeout: per-query (= NSGA-II mean time for that query)")

        # Clear caches before DP so it doesn't benefit from NSGA-II's evaluations
        clear_evaluation_caches()
        gc.collect()

        # ------------------------------------------------------------------
        # Step 2: Run DP no pruning (baseline + HV reference bounds)
        # ------------------------------------------------------------------
        dp_base_results = {}
        baselines = {}  # {query: {hv, g_mins, g_maxs}}
        dp_cache_file = Path('results') / f'dp_cache_k{k}.pkl'

        if args.reuse_dp and dp_cache_file.exists():
            print(f"\n--- DP No Pruning (k={k}) [CACHED] ---")
            with open(dp_cache_file, 'rb') as f:
                cache = pickle.load(f)
            dp_base_results = cache['dp_base']
            baselines = cache['baselines']
            dp_delta_results = cache['dp_delta']
            for q in QUERIES:
                if q not in dp_base_results:
                    # Run DP for queries not in cache
                    print(f"  {q} ... [not cached, running]", end=" ", flush=True)
                    sols, dt = run_timed(
                        FPTAS, q, df_history, k,
                        epsilon=0.0, disable_pruning=True,
                        pruning_strategy=None, verbose=False
                    )
                    _, g_mins, g_maxs = compute_hypervolume(sols, return_bounds=True)
                    pareto = extract_pareto_front(sols)
                    hv = compute_hypervolume(pareto, g_mins, g_maxs)
                    baselines[q] = {'hv': hv, 'g_mins': g_mins, 'g_maxs': g_maxs}
                    dp_base_results[q] = (sols, pareto, dt, hv)
                    print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s")
                else:
                    sols, pareto, dt, hv = dp_base_results[q]
                    print(f"  {q} ... {len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s [cached]")

            print(f"\n--- DP + Delta Pruning (delta={DELTA}, k={k}) [CACHED] ---")
            for q in QUERIES:
                if q not in dp_delta_results:
                    print(f"  {q} ... [not cached, running]", end=" ", flush=True)
                    sols, dt = run_timed(
                        FPTAS, q, df_history, k,
                        delta=DELTA, disable_pruning=False,
                        pruning_strategy='delta_qoa', verbose=False
                    )
                    pareto = extract_pareto_front(sols)
                    hv = compute_hypervolume(pareto, baselines[q]['g_mins'], baselines[q]['g_maxs'])
                    dp_delta_results[q] = (sols, pareto, dt, hv)
                    print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s")
                else:
                    sols, pareto, dt, hv = dp_delta_results[q]
                    print(f"  {q} ... {len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s [cached]")

            # Re-save cache with any new queries added
            dp_cache_file.parent.mkdir(exist_ok=True)
            with open(dp_cache_file, 'wb') as f:
                pickle.dump({
                    'dp_base': dp_base_results,
                    'baselines': baselines,
                    'dp_delta': dp_delta_results,
                }, f)
        else:
            print(f"\n--- DP No Pruning (k={k}) ---")
            for q in QUERIES:
                print(f"  {q} ...", end=" ", flush=True)
                sols, dt = run_timed(
                    FPTAS, q, df_history, k,
                    epsilon=0.0, disable_pruning=True,
                    pruning_strategy=None, verbose=False
                )
                _, g_mins, g_maxs = compute_hypervolume(sols, return_bounds=True)
                pareto = extract_pareto_front(sols)
                hv = compute_hypervolume(pareto, g_mins, g_maxs)
                baselines[q] = {'hv': hv, 'g_mins': g_mins, 'g_maxs': g_maxs}
                dp_base_results[q] = (sols, pareto, dt, hv)
                print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s")

            # Clear caches before DP+Delta so it doesn't benefit from DP's evaluations
            clear_evaluation_caches()
            gc.collect()

            # ------------------------------------------------------------------
            # Step 3: Run DP + delta pruning
            # ------------------------------------------------------------------
            dp_delta_results = {}

            print(f"\n--- DP + Delta Pruning (delta={DELTA}, k={k}) ---")
            for q in QUERIES:
                print(f"  {q} ...", end=" ", flush=True)
                sols, dt = run_timed(
                    FPTAS, q, df_history, k,
                    delta=DELTA, disable_pruning=False,
                    pruning_strategy='delta_qoa', verbose=False
                )
                pareto = extract_pareto_front(sols)
                hv = compute_hypervolume(pareto, baselines[q]['g_mins'], baselines[q]['g_maxs'])
                dp_delta_results[q] = (sols, pareto, dt, hv)
                print(f"{len(sols)} sols, {len(pareto)} Pareto, HV={hv:.6f}, {dt:.1f}s")

            # Save DP cache
            dp_cache_file.parent.mkdir(exist_ok=True)
            with open(dp_cache_file, 'wb') as f:
                pickle.dump({
                    'dp_base': dp_base_results,
                    'baselines': baselines,
                    'dp_delta': dp_delta_results,
                }, f)

        # Compute DP reference Pareto front in minimization space for IGD
        for q in QUERIES:
            dp_pareto = dp_base_results[q][1]  # pareto list from (sols, pareto, dt, hv)
            g_mins, g_maxs = baselines[q]['g_mins'], baselines[q]['g_maxs']
            raw = []
            for s in dp_pareto:
                if hasattr(s, 'metrics') and s.metrics is not None:
                    raw.append(list(s.metrics))
                else:
                    raw.append([s.cost, s.latency, s.energy, s.qoa])
            raw = np.array(raw)
            ranges = g_maxs - g_mins
            ranges = np.where(ranges == 0, 1.0, ranges)
            norm = (raw - g_mins) / ranges
            norm[:, 0] = 1.0 - norm[:, 0]
            norm[:, 1] = 1.0 - norm[:, 1]
            norm[:, 2] = 1.0 - norm[:, 2]
            baselines[q]['ref_pareto_min'] = 1.0 - norm  # minimization space

        # Clear caches before MOQO so it doesn't benefit from DP's evaluations
        clear_evaluation_caches()
        gc.collect()

        # ------------------------------------------------------------------
        # Step 4: Run MOQO no pruning (NUM_RUNS times, timeout = max NSGA-II mean time)
        # ------------------------------------------------------------------
        moqo_base_run_data = {}  # {query: [list of (sols, dt) per run]}

        print(f"\n--- MOQO No Pruning (per-query timeout, k={k}, {NUM_RUNS} runs) ---")
        for q in QUERIES:
            q_timeout = moqo_timeout_per_query[q]
            moqo_base_run_data[q] = []
            for run_id in range(NUM_RUNS):
                # Clear caches and re-seed RNG so each run is independent
                clear_evaluation_caches()
                gc.collect()
                random.seed(run_id)
                np.random.seed(run_id)
                print(f"  {q} [run {run_id+1}/{NUM_RUNS}] (timeout={q_timeout:.1f}s) ...", end=" ", flush=True)
                sols, dt = run_timed(
                    RandomMOQO, q, df_history,
                    timeout_seconds=q_timeout, max_nodes=k,
                    pruning_strategy='none', verbose=False
                )
                moqo_base_run_data[q].append((sols, dt))
                print(f"{len(sols)} sols, {dt:.1f}s")

        # ------------------------------------------------------------------
        # Step 5: Run MOQO + delta pruning (NUM_RUNS times)
        # ------------------------------------------------------------------
        moqo_delta_run_data = {}  # {query: [list of (sols, dt) per run]}

        print(f"\n--- MOQO + Delta Pruning (delta={DELTA}, per-query timeout, k={k}, {NUM_RUNS} runs) ---")
        for q in QUERIES:
            q_timeout = moqo_timeout_per_query[q]
            moqo_delta_run_data[q] = []
            for run_id in range(NUM_RUNS):
                # Clear caches and re-seed RNG so each run is independent
                clear_evaluation_caches()
                gc.collect()
                random.seed(run_id)
                np.random.seed(run_id)
                print(f"  {q} [run {run_id+1}/{NUM_RUNS}] (timeout={q_timeout:.1f}s) ...", end=" ", flush=True)
                sols, dt = run_timed(
                    RandomMOQO, q, df_history,
                    timeout_seconds=q_timeout, max_nodes=k,
                    pruning_strategy='delta_qoa', delta=DELTA, verbose=False
                )
                moqo_delta_run_data[q].append((sols, dt))
                print(f"{len(sols)} sols, {dt:.1f}s")

        # ------------------------------------------------------------------
        # Compute per-run metrics using DP baseline bounds
        # ------------------------------------------------------------------
        def compute_run_metrics(run_data_dict, algo_name):
            """Compute per-run metrics for a stochastic algorithm.
            run_data_dict: {query: [(sols, dt), ...]}
            Returns: {query: [{'sols':..., 'pareto':..., 'dt':..., 'hv':..., 'hv_ret':..., 'qoa':..., 'igd':...}, ...]}
            """
            result = {}
            for q in QUERIES:
                result[q] = []
                b_hv = baselines[q]['hv']
                for sols, dt in run_data_dict[q]:
                    pareto = extract_pareto_front(sols)
                    hv = compute_hypervolume(pareto, baselines[q]['g_mins'], baselines[q]['g_maxs'])
                    hv_ret = (hv / b_hv * 100) if b_hv > 0 else 0.0
                    qoa = get_best_qoa(sols)
                    igd_val = compute_igd(sols, baselines[q]['ref_pareto_min'],
                                         baselines[q]['g_mins'], baselines[q]['g_maxs'])
                    result[q].append({
                        'sols': len(sols), 'pareto': len(pareto),
                        'dt': dt, 'hv': hv,
                        'hv_ret': hv_ret, 'qoa': qoa, 'igd': igd_val,
                    })
            return result

        nsga_metrics = compute_run_metrics(nsga_run_data, 'NSGA-II')
        moqo_base_metrics = compute_run_metrics(moqo_base_run_data, 'MOQO')
        moqo_delta_metrics = compute_run_metrics(moqo_delta_run_data, 'MOQO+Delta')

        # ------------------------------------------------------------------
        # Filter outlier runs (time > 2x expected timeout)
        # ------------------------------------------------------------------
        def filter_outliers(run_metrics_dict, timeout_dict, algo_label):
            """Remove runs where time > 2x per-query timeout (stuck iteration)."""
            for q in QUERIES:
                threshold = timeout_dict[q] * 2.0
                original = run_metrics_dict[q]
                filtered = [r for r in original if r['dt'] <= threshold]
                n_removed = len(original) - len(filtered)
                if n_removed > 0:
                    print(f"  WARNING: {algo_label} {q}: removed {n_removed}/{len(original)} "
                          f"outlier run(s) (time > {threshold:.1f}s)")
                # Keep at least 1 run even if all are outliers
                run_metrics_dict[q] = filtered if filtered else [min(original, key=lambda r: r['dt'])]
            return run_metrics_dict

        moqo_base_metrics = filter_outliers(moqo_base_metrics, moqo_timeout_per_query, 'MOQO')
        moqo_delta_metrics = filter_outliers(moqo_delta_metrics, moqo_timeout_per_query, 'MOQO+Delta')

        # ------------------------------------------------------------------
        # Aggregate per-query results for this k
        # ------------------------------------------------------------------
        def aggregate_stochastic(run_metrics_q):
            """Aggregate list of run dicts into mean/std entry."""
            keys = ['dt', 'sols', 'pareto', 'hv', 'hv_ret', 'qoa', 'igd']
            field_map = {'dt': 'time', 'sols': 'sols',
                         'pareto': 'pareto', 'hv': 'hv', 'hv_ret': 'hv_ret',
                         'qoa': 'qoa', 'igd': 'igd'}
            entry = {'runs': run_metrics_q}
            for k_name in keys:
                vals = [r[k_name] for r in run_metrics_q]
                f = field_map[k_name]
                entry[f'{f}_mean'] = float(np.mean(vals))
                entry[f'{f}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return entry

        def aggregate_deterministic(data_tuple, query):
            """Create entry from a single deterministic run tuple."""
            sols_list, pareto_list, dt, hv_val = data_tuple
            b_hv_local = baselines[query]['hv']
            hv_ret = (hv_val / b_hv_local * 100) if b_hv_local > 0 else 0.0
            qoa = get_best_qoa(sols_list)
            igd_val = compute_igd(sols_list, baselines[query]['ref_pareto_min'],
                                 baselines[query]['g_mins'], baselines[query]['g_maxs'])
            return {
                'time_mean': dt, 'time_std': 0.0,
                'sols_mean': len(sols_list), 'sols_std': 0.0,
                'pareto_mean': len(pareto_list), 'pareto_std': 0.0,
                'hv_mean': hv_val, 'hv_std': 0.0,
                'hv_ret_mean': hv_ret, 'hv_ret_std': 0.0,
                'qoa_mean': qoa, 'qoa_std': 0.0,
                'igd_mean': igd_val, 'igd_std': 0.0,
                'runs': [{'sols': len(sols_list), 'pareto': len(pareto_list),
                          'dt': dt, 'hv': hv_val,
                          'hv_ret': hv_ret, 'qoa': qoa, 'igd': igd_val}],
            }

        for q in QUERIES:
            all_results[k][q] = {
                'NSGA-II':    aggregate_stochastic(nsga_metrics[q]),
                'DP':         aggregate_deterministic(dp_base_results[q], q),
                'DP+Delta':   aggregate_deterministic(dp_delta_results[q], q),
                'MOQO':       aggregate_stochastic(moqo_base_metrics[q]),
                'MOQO+Delta': aggregate_stochastic(moqo_delta_metrics[q]),
            }
            # DP is the IGD reference set — 0.0 by definition
            all_results[k][q]['DP']['igd_mean'] = 0.0
            all_results[k][q]['DP']['runs'][0]['igd'] = 0.0

        # ------------------------------------------------------------------
        # Print per-k summary table
        # ------------------------------------------------------------------
        algos = ['NSGA-II', 'DP', 'DP+Delta', 'MOQO', 'MOQO+Delta']

        print(f"\n\n{'='*120}")
        print(f"SUMMARY k={k} (avg across {len(QUERIES)} queries, MOQO timeout=per-query, median={moqo_timeout_median:.1f}s)")
        print(f"{'='*120}\n")

        print(f"{'Algorithm':<16} {'Time(s)':<20} {'Pareto':<18} {'HV%':<18} {'IGD':<18}")
        print("-" * 98)

        for algo in algos:
            t_means = [all_results[k][q][algo]['time_mean'] for q in QUERIES]
            t_stds  = [all_results[k][q][algo]['time_std'] for q in QUERIES]
            p_means = [all_results[k][q][algo]['pareto_mean'] for q in QUERIES]
            p_stds  = [all_results[k][q][algo]['pareto_std'] for q in QUERIES]
            h_means = [all_results[k][q][algo]['hv_ret_mean'] for q in QUERIES]
            h_stds  = [all_results[k][q][algo]['hv_ret_std'] for q in QUERIES]
            i_means = [all_results[k][q][algo]['igd_mean'] for q in QUERIES]
            i_stds  = [all_results[k][q][algo]['igd_std'] for q in QUERIES]

            avg_t = np.mean(t_means)
            avg_t_std = np.mean(t_stds) if algo in STOCHASTIC_ALGOS else None
            avg_p = np.mean(p_means)
            avg_p_std = np.mean(p_stds) if algo in STOCHASTIC_ALGOS else None
            avg_h = np.mean(h_means)
            avg_h_std = np.mean(h_stds) if algo in STOCHASTIC_ALGOS else None
            avg_i = np.mean(i_means)
            avg_i_std = np.mean(i_stds) if algo in STOCHASTIC_ALGOS else None

            print(f"{algo:<16} {fmt_mean_std(avg_t, avg_t_std):<20} "
                  f"{fmt_mean_std(avg_p, avg_p_std, 0):<18} "
                  f"{fmt_mean_std(avg_h, avg_h_std):<18} "
                  f"{fmt_mean_std(avg_i, avg_i_std, 4):<18}")

    # ======================================================================
    # Grand summary across all k
    # ======================================================================
    algos = ['NSGA-II', 'DP', 'DP+Delta', 'MOQO', 'MOQO+Delta']

    print(f"\n\n{'#'*120}")
    print(f"#  GRAND SUMMARY (avg across {len(QUERIES)} queries, {NUM_RUNS} runs for stochastic)")
    print(f"{'#'*120}\n")

    print(f"{'k':<5} {'Algorithm':<16} {'Time(s)':<20} {'Pareto':<18} {'HV%':<18} {'IGD':<18}")
    print("-" * 98)

    for k in K_VALUES:
        for algo in algos:
            t_means = [all_results[k][q][algo]['time_mean'] for q in QUERIES]
            t_stds  = [all_results[k][q][algo]['time_std'] for q in QUERIES]
            p_means = [all_results[k][q][algo]['pareto_mean'] for q in QUERIES]
            p_stds  = [all_results[k][q][algo]['pareto_std'] for q in QUERIES]
            h_means = [all_results[k][q][algo]['hv_ret_mean'] for q in QUERIES]
            h_stds  = [all_results[k][q][algo]['hv_ret_std'] for q in QUERIES]
            i_means = [all_results[k][q][algo]['igd_mean'] for q in QUERIES]
            i_stds  = [all_results[k][q][algo]['igd_std'] for q in QUERIES]

            avg_t = np.mean(t_means)
            avg_t_std = np.mean(t_stds) if algo in STOCHASTIC_ALGOS else None
            avg_p = np.mean(p_means)
            avg_p_std = np.mean(p_stds) if algo in STOCHASTIC_ALGOS else None
            avg_h = np.mean(h_means)
            avg_h_std = np.mean(h_stds) if algo in STOCHASTIC_ALGOS else None
            avg_i = np.mean(i_means)
            avg_i_std = np.mean(i_stds) if algo in STOCHASTIC_ALGOS else None

            print(f"{k:<5} {algo:<16} {fmt_mean_std(avg_t, avg_t_std):<20} "
                  f"{fmt_mean_std(avg_p, avg_p_std, 0):<18} "
                  f"{fmt_mean_std(avg_h, avg_h_std):<18} "
                  f"{fmt_mean_std(avg_i, avg_i_std, 4):<18}")
        print()

    # ======================================================================
    # Save results
    # ======================================================================
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'algorithm_comparison_k1to5.csv'

    rows = []
    for k in K_VALUES:
        for q in QUERIES:
            for algo in algos:
                entry = all_results[k][q][algo]
                # Write individual run rows
                for run_id, run in enumerate(entry['runs']):
                    rows.append({
                        'k': k, 'query': q, 'algorithm': algo,
                        'run_id': run_id,
                        'time_s': run['dt'],
                        'solutions': run['sols'], 'pareto': run['pareto'],
                        'hv': run['hv'], 'hv_retention_pct': run['hv_ret'],
                        'best_qoa': run['qoa'], 'igd': run['igd'],
                        'time_std': None,
                        'sols_std': None, 'pareto_std': None,
                        'hv_std': None, 'hv_ret_std': None,
                        'qoa_std': None, 'igd_std': None,
                    })
                # Write averaged summary row
                rows.append({
                    'k': k, 'query': q, 'algorithm': algo,
                    'run_id': 'mean',
                    'time_s': entry['time_mean'],
                    'solutions': entry['sols_mean'], 'pareto': entry['pareto_mean'],
                    'hv': entry['hv_mean'], 'hv_retention_pct': entry['hv_ret_mean'],
                    'best_qoa': entry['qoa_mean'], 'igd': entry['igd_mean'],
                    'time_std': entry['time_std'],
                    'sols_std': entry['sols_std'], 'pareto_std': entry['pareto_std'],
                    'hv_std': entry['hv_std'], 'hv_ret_std': entry['hv_ret_std'],
                    'qoa_std': entry['qoa_std'], 'igd_std': entry['igd_std'],
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_file, index=False)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
