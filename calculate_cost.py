#!/usr/bin/env python3
"""
Re-evaluate estimated_cost/latency/energy/qoa in-place for CSV files.

Each row's `level` column determines which level_X_data.csv is used.

Usage:
    # Update a CSV in-place:
    python calculate_cost.py /Users/aamirhamid/Downloads/DP_Moqo/DP_Delta.csv

    # Multiple files:
    python calculate_cost.py /Users/aamirhamid/Downloads/DP_Moqo/DP_Delta.csv /Users/aamirhamid/Downloads/DP_Moqo/MOQO.csv

    # Dry run (print changes, don't write):
    python calculate_cost.py --dry-run file.csv
"""

import argparse
import importlib.util
import re
import sys
import time
import types

import pandas as pd


def load_old_main():
    old_pkg = types.ModuleType("old")
    old_pkg.__path__ = ["old"]
    old_pkg.__package__ = "old"
    sys.modules["old"] = old_pkg

    spec = importlib.util.spec_from_file_location(
        "old.main", "old/main.py", submodule_search_locations=[]
    )
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "old"
    sys.modules["old.main"] = m
    spec.loader.exec_module(m)
    return m


def parse_assignment(s):
    return tuple(int(x) for x in re.findall(r"\d+", s))


def load_level_data(level):
    path = f"data/raw/levels/level_{level}_data.csv"
    df = pd.read_csv(path)
    df["llm_assignments"] = (
        df["llm_assignments"]
        .astype(str)
        .str.replace(r"[\(\)\s]", "", regex=True)
        .str.rstrip(",")
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate estimated_cost/latency/energy/qoa in CSV files in-place."
    )
    parser.add_argument("files", nargs="+", help="CSV file(s) to update")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--query-tokens", type=int, default=215)
    parser.add_argument("--blend-tokens", type=int, default=26)
    parser.add_argument("--ctx-tokens", type=int, default=39)
    args = parser.parse_args()

    m = load_old_main()

    # Pre-load all level data
    level_data = {}
    for lvl in range(5):
        try:
            level_data[lvl] = load_level_data(lvl)
        except FileNotFoundError:
            print(f"WARNING: level_{lvl}_data.csv not found, skipping level {lvl}")

    for filepath in args.files:
        print(f"\n{'='*80}")
        print(f"Processing: {filepath}")
        print(f"{'='*80}")

        df = pd.read_csv(filepath)
        total = len(df)
        changed_count = 0

        # Fix NaN query_type: fall back to query_type_norm or Query Type
        nan_before = df["query_type"].isna().sum()
        if nan_before > 0:
            for fallback_col in ["query_type_norm", "Query Type"]:
                if fallback_col in df.columns:
                    mask = df["query_type"].isna()
                    df.loc[mask, "query_type"] = df.loc[mask, fallback_col]
            nan_after = df["query_type"].isna().sum()
            print(f"NaN query_type: {nan_before} found, {nan_before - nan_after} recovered, {nan_after} remaining")

        # Deduplicate evaluations: group by (level, struct_id, assignment, query_type)
        eval_cache = {}
        unique_keys = df[["level", "struct_id", "assignment", "query_type"]].drop_duplicates()
        n_unique = len(unique_keys)
        print(f"Total rows: {total}, unique evaluations: {n_unique}")

        t0 = time.time()
        for i, (_, urow) in enumerate(unique_keys.iterrows()):
            level = int(urow["level"])
            if level not in level_data:
                continue

            struct_id = int(urow["struct_id"])
            assignment = parse_assignment(urow["assignment"])
            query_type = urow["query_type"]

            if pd.isna(query_type):
                continue

            cost, lat, energy, qoa = m.evaluate_individual_V2(
                struct_id, assignment, query_type,
                args.query_tokens, args.blend_tokens, args.ctx_tokens,
                level_data[level],
            )
            eval_cache[(level, struct_id, urow["assignment"], query_type)] = (
                cost, lat, energy, float(qoa)
            )

            if (i + 1) % 100 == 0 or i + 1 == n_unique:
                elapsed = time.time() - t0
                print(f"  Evaluated {i+1}/{n_unique} unique combos ({elapsed:.1f}s)")

        # Apply results back to dataframe
        for idx, row in df.iterrows():
            key = (int(row["level"]), int(row["struct_id"]), row["assignment"], row["query_type"])
            if key not in eval_cache:
                continue

            new_cost, new_lat, new_energy, new_qoa = eval_cache[key]
            old_cost = row["estimated_cost"]
            old_lat = row["estimated_latency"]
            old_energy = row["estimated_energy"]
            old_qoa = row["estimated_qoa"]

            diff = (
                abs(new_cost - old_cost) > 1e-12
                or abs(new_lat - old_lat) > 1e-6
                or abs(new_energy - old_energy) > 1e-6
                or abs(new_qoa - old_qoa) > 1e-6
            )

            if diff:
                changed_count += 1
                if args.dry_run and changed_count <= 20:
                    print(f"  Row {idx}: level={key[0]} struct={key[1]} assign={key[2]} query={key[3]}")
                    print(f"    cost:    {old_cost:.6e} -> {new_cost:.6e}")
                    print(f"    latency: {old_lat:.3f} -> {new_lat:.3f}")
                    print(f"    energy:  {old_energy:.3f} -> {new_energy:.3f}")
                    print(f"    qoa:     {old_qoa:.6f} -> {new_qoa:.6f}")

            df.at[idx, "estimated_cost"] = new_cost
            df.at[idx, "estimated_latency"] = new_lat
            df.at[idx, "estimated_energy"] = new_energy
            df.at[idx, "estimated_qoa"] = new_qoa

        elapsed = time.time() - t0
        print(f"\nRows changed: {changed_count}/{total} ({elapsed:.1f}s)")

        if args.dry_run:
            if changed_count > 20:
                print(f"  ... and {changed_count - 20} more (use without --dry-run to apply)")
            print("DRY RUN — no file written.")
        else:
            df.to_csv(filepath, index=False)
            print(f"Written: {filepath}")


if __name__ == "__main__":
    main()
