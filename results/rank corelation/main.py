import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    0: "level_0_mmlu.csv",
    1: "level_1_mmlu.csv",
    2: "level_2_mmlu.csv",
    3: "level_3_mmlu.csv",
    4: "level_4_mmlu.csv",
}

OUTPUT_DIR = "mmlu_rank_pareto_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

QUESTION_COL = "Original_Query"

# Estimated metrics
EST_QOA = "qoa"
EST_COST = "cost"
EST_LATENCY = "latency"
EST_ENERGY = "energy"

# Realized metrics
REAL_QOA = "New_QoA_Score"
REAL_COST = "Total_Cost"
REAL_LATENCY = "Total_Latency"
REAL_ENERGY = "Total_Energy"


# ============================================================
# PARETO FUNCTION
# ============================================================

def pareto_front(
    df,
    qoa_col,
    cost_col,
    latency_col,
    energy_col
):
    """
    Return the indices of nondominated candidate plans.

    Objectives:
        QoA     -> maximize
        Cost    -> minimize
        Latency -> minimize
        Energy  -> minimize

    Plan B dominates Plan A if:
      - B is at least as good as A on all four objectives
      - B is strictly better on at least one objective
    """

    cols = [
        qoa_col,
        cost_col,
        latency_col,
        energy_col
    ]

    values = df[cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Only compare rows having all four metrics
    valid_mask = values.notna().all(axis=1)

    valid_indices = list(
        df.index[valid_mask]
    )

    if len(valid_indices) == 0:
        return set()

    array = values.loc[valid_indices].to_numpy(
        dtype=float
    )

    pareto = set()

    for i, idx_i in enumerate(valid_indices):

        plan_a = array[i]

        dominated = False

        for j, idx_j in enumerate(valid_indices):

            if i == j:
                continue

            plan_b = array[j]

            # B is at least as good as A
            at_least_as_good = (
                plan_b[0] >= plan_a[0] and
                plan_b[1] <= plan_a[1] and
                plan_b[2] <= plan_a[2] and
                plan_b[3] <= plan_a[3]
            )

            # B is strictly better in at least one dimension
            strictly_better = (
                plan_b[0] > plan_a[0] or
                plan_b[1] < plan_a[1] or
                plan_b[2] < plan_a[2] or
                plan_b[3] < plan_a[3]
            )

            if at_least_as_good and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto.add(idx_i)

    return pareto


# ============================================================
# STORAGE
# ============================================================

rank_results = []
pareto_results = []


# ============================================================
# PROCESS EACH PERFDB LEVEL
# ============================================================

for level, filename in FILES.items():

    print("\n" + "=" * 70)
    print(f"Processing PerfDB Level {level}")
    print("=" * 70)

    df = pd.read_csv(filename)

    # --------------------------------------------------------
    # Convert metrics to numeric
    # --------------------------------------------------------

    numeric_columns = [
        EST_QOA,
        EST_COST,
        EST_LATENCY,
        EST_ENERGY,
        REAL_QOA,
        REAL_COST,
        REAL_LATENCY,
        REAL_ENERGY,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Estimated QoA is bounded to [0,1]
    # --------------------------------------------------------

    df["estimated_qoa"] = df[EST_QOA].clip(
        lower=0,
        upper=1
    )

    # Cost/latency/energy are NOT clipped.
    df["estimated_cost"] = df[EST_COST]
    df["estimated_latency"] = df[EST_LATENCY]
    df["estimated_energy"] = df[EST_ENERGY]

    df["realized_qoa"] = df[REAL_QOA]
    df["realized_cost"] = df[REAL_COST]
    df["realized_latency"] = df[REAL_LATENCY]
    df["realized_energy"] = df[REAL_ENERGY]

    print(
        "Rows:",
        len(df),
        "| Questions:",
        df[QUESTION_COL].nunique()
    )

    # ========================================================
    # ANALYSIS 1: PER-QUERY SPEARMAN RANK CORRELATION
    # ========================================================

    metric_pairs = {
        "QoA": (
            "estimated_qoa",
            "realized_qoa"
        ),
        "Cost": (
            "estimated_cost",
            "realized_cost"
        ),
        "Latency": (
            "estimated_latency",
            "realized_latency"
        ),
        "Energy": (
            "estimated_energy",
            "realized_energy"
        ),
    }

    for question, group in df.groupby(
        QUESTION_COL,
        sort=False
    ):

        for metric, (
            estimated_col,
            realized_col
        ) in metric_pairs.items():

            g = group.dropna(
                subset=[
                    estimated_col,
                    realized_col
                ]
            )

            rho = np.nan
            valid = False

            # Need at least two candidates
            # AND variation on both sides
            if (
                len(g) >= 2
                and g[estimated_col].nunique() >= 2
                and g[realized_col].nunique() >= 2
            ):

                rho, pvalue = spearmanr(
                    g[estimated_col],
                    g[realized_col]
                )

                if np.isfinite(rho):
                    valid = True

            rank_results.append({
                "level": level,
                "question": question,
                "metric": metric,
                "n_candidates": len(group),
                "valid_spearman": valid,
                "spearman_rho": rho,
            })

    # ========================================================
    # ANALYSIS 2: PARETO-FRONT AGREEMENT
    # ========================================================

    for question, group in df.groupby(
        QUESTION_COL,
        sort=False
    ):

        group = group.copy()

        # ----------------------------------------------------
        # Estimated Pareto frontier
        # ----------------------------------------------------

        estimated_front = pareto_front(
            group,
            qoa_col="estimated_qoa",
            cost_col="estimated_cost",
            latency_col="estimated_latency",
            energy_col="estimated_energy"
        )

        # ----------------------------------------------------
        # Realized Pareto frontier
        # ----------------------------------------------------

        realized_front = pareto_front(
            group,
            qoa_col="realized_qoa",
            cost_col="realized_cost",
            latency_col="realized_latency",
            energy_col="realized_energy"
        )

        intersection = (
            estimated_front &
            realized_front
        )

        # ----------------------------------------------------
        # Pareto Precision
        #
        # Of plans predicted Pareto-optimal,
        # how many were actually Pareto-optimal?
        # ----------------------------------------------------

        if len(estimated_front) > 0:
            pareto_precision = (
                len(intersection) /
                len(estimated_front)
            )
        else:
            pareto_precision = np.nan

        # ----------------------------------------------------
        # Pareto Recall
        #
        # Of actually Pareto-optimal plans,
        # how many were preserved?
        # ----------------------------------------------------

        if len(realized_front) > 0:
            pareto_recall = (
                len(intersection) /
                len(realized_front)
            )
        else:
            pareto_recall = np.nan

        # ====================================================
        # QOA-SPECIFIC RETENTION
        # ====================================================

        # MMLU-Pro realized QoA is binary.
        #
        # Find all correct candidates.
        correct_candidates = set(
            group.index[
                group["realized_qoa"] > 0
            ]
        )

        has_correct_candidate = (
            len(correct_candidates) > 0
        )

        # ----------------------------------------------------
        # Did estimated Pareto frontier retain
        # AT LEAST ONE correct candidate?
        # ----------------------------------------------------

        if has_correct_candidate:

            correct_on_front = (
                estimated_front &
                correct_candidates
            )

            retained_at_least_one_correct = (
                len(correct_on_front) > 0
            )

            # Fraction of all correct candidates retained
            correct_candidate_recall = (
                len(correct_on_front) /
                len(correct_candidates)
            )

        else:

            retained_at_least_one_correct = np.nan
            correct_candidate_recall = np.nan

        pareto_results.append({

            "level": level,

            "question": question,

            "n_candidates":
                len(group),

            "estimated_front_size":
                len(estimated_front),

            "realized_front_size":
                len(realized_front),

            "front_overlap_size":
                len(intersection),

            "pareto_precision":
                pareto_precision,

            "pareto_recall":
                pareto_recall,

            "has_correct_candidate":
                has_correct_candidate,

            "retained_at_least_one_correct":
                retained_at_least_one_correct,

            "correct_candidate_recall":
                correct_candidate_recall,
        })


# ============================================================
# CREATE DATAFRAMES
# ============================================================

rank_df = pd.DataFrame(rank_results)
pareto_df = pd.DataFrame(pareto_results)


# ============================================================
# RANK CORRELATION SUMMARY
# ============================================================

valid_rank = rank_df[
    rank_df["valid_spearman"] == True
].copy()

rank_summary = (
    valid_rank
    .groupby(
        ["level", "metric"]
    )
    .agg(
        valid_questions=(
            "spearman_rho",
            "count"
        ),
        mean_spearman=(
            "spearman_rho",
            "mean"
        ),
        median_spearman=(
            "spearman_rho",
            "median"
        ),
        std_spearman=(
            "spearman_rho",
            "std"
        ),
    )
    .reset_index()
)


# ============================================================
# PARETO SUMMARY
# ============================================================

pareto_summary = (
    pareto_df
    .groupby("level")
    .agg(

        questions=(
            "question",
            "count"
        ),

        mean_pareto_precision=(
            "pareto_precision",
            "mean"
        ),

        mean_pareto_recall=(
            "pareto_recall",
            "mean"
        ),

        mean_estimated_front_size=(
            "estimated_front_size",
            "mean"
        ),

        mean_realized_front_size=(
            "realized_front_size",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# QOA RETENTION SUMMARY
# ============================================================

qoa_retention = (
    pareto_df[
        pareto_df["has_correct_candidate"] == True
    ]
    .groupby("level")
    .agg(

        queries_with_correct_candidate=(
            "question",
            "count"
        ),

        correct_plan_retention_rate=(
            "retained_at_least_one_correct",
            "mean"
        ),

        mean_correct_candidate_recall=(
            "correct_candidate_recall",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE RESULTS
# ============================================================

rank_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rank_per_query.csv"
    ),
    index=False
)

rank_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rank_summary.csv"
    ),
    index=False
)

pareto_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pareto_per_query.csv"
    ),
    index=False
)

pareto_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pareto_summary.csv"
    ),
    index=False
)

qoa_retention.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "qoa_pareto_retention_summary.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MEAN PER-QUERY SPEARMAN RANK CORRELATION")
print("=" * 70)

rank_table = rank_summary.pivot(
    index="level",
    columns="metric",
    values="mean_spearman"
)

# Desired ordering
column_order = [
    "QoA",
    "Cost",
    "Latency",
    "Energy"
]

rank_table = rank_table[
    column_order
]

print(
    rank_table.round(3).to_string()
)


print("\n")
print("=" * 70)
print("VALID QUESTIONS FOR SPEARMAN")
print("=" * 70)

valid_table = rank_summary.pivot(
    index="level",
    columns="metric",
    values="valid_questions"
)

valid_table = valid_table[
    column_order
]

print(
    valid_table.to_string()
)


print("\n")
print("=" * 70)
print("PARETO-FRONT AGREEMENT")
print("=" * 70)

print(
    pareto_summary[
        [
            "level",
            "mean_pareto_precision",
            "mean_pareto_recall",
            "mean_estimated_front_size",
            "mean_realized_front_size",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print("\n")
print("=" * 70)
print("QOA RETENTION ON ESTIMATED PARETO FRONT")
print("=" * 70)

print(
    qoa_retention
    .round(3)
    .to_string(index=False)
)


print("\nResults written to:", OUTPUT_DIR)