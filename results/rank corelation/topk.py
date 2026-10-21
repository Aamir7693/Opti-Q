from pathlib import Path
import pandas as pd
import numpy as np

files = {
    0: "level_0_mmlu.csv",
    1: "level_1_mmlu.csv",
    2: "level_2_mmlu.csv",
    3: "level_3_mmlu.csv",
    4: "level_4_mmlu.csv",
}


rows = []

for level, path in files.items():
    df = pd.read_csv(path).copy()
    for c in ["qoa", "New_QoA_Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["estimated_qoa"] = df["qoa"].clip(0, 1)

    for question, g in df.groupby("Original_Query", sort=False):
        g = g.copy()

        # User's actual selection rule: choose the FIRST retained plan
        selected = g.iloc[0]

        best_realized = g["New_QoA_Score"].max()
        selected_realized = selected["New_QoA_Score"]

        rows.append({
            "level": level,
            "question": question,
            "selected_estimated_qoa": selected["estimated_qoa"],
            "selected_realized_qoa": selected_realized,
            "best_realized_qoa_in_top5": best_realized,
            "top5_hit": float(np.isclose(selected_realized, best_realized)),
            "top5_regret": float(best_realized - selected_realized),
            "any_correct_in_top5": int((g["New_QoA_Score"] > 0).any()),
            "selected_correct": int(selected_realized > 0),
        })

res = pd.DataFrame(rows)

summary = (
    res.groupby("level")
    .agg(
        questions=("question", "count"),
        top5_hit_rate=("top5_hit", "mean"),
        mean_top5_regret=("top5_regret", "mean"),
        questions_with_correct_candidate=("any_correct_in_top5", "sum"),
    )
    .reset_index()
)

conditional = (
    res[res["any_correct_in_top5"] == 1]
    .groupby("level")
    .agg(
        selected_correct_when_correct_available=("selected_correct", "mean")
    )
    .reset_index()
)

summary = summary.merge(conditional, on="level", how="left")
summary.to_csv("mmlu_top5_first_plan_summary.csv", index=False)
res.to_csv("mmlu_top5_first_plan_per_query.csv", index=False)

print(summary.round(4).to_string(index=False))
print("\nSaved:")
print("/mnt/data/mmlu_top5_first_plan_summary.csv")
print("/mnt/data/mmlu_top5_first_plan_per_query.csv")
