#!/usr/bin/env python3
"""Quick test: FPTAS with k=4"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.llm_dag_optimizer.core import FPTAS

print("="*80)
print("Quick Test: FPTAS with k=4")
print("="*80)

df = pd.read_csv("data/raw/levels/level_2_data.csv")

solutions = FPTAS(
    query_type="Art",
    df_history=df,
    max_nodes=4,
    epsilon=0.05,
    verbose=True,
    disable_pruning=False,
)

print(f"\n✅ SUCCESS! FPTAS works with k=4")
print(f"   Solutions found: {len(solutions)}")
print(f"   QoA range: [{min(s.qoa for s in solutions):.4f}, {max(s.qoa for s in solutions):.4f}]")
print(f"   Cost range: [{min(s.cost for s in solutions):.6f}, {max(s.cost for s in solutions):.6f}]")

best = max(solutions, key=lambda s: s.qoa)
print(f"\n   Best solution:")
print(f"     QoA={best.qoa:.4f}, Cost={best.cost:.6f}")
print(f"     Struct={best.struct_id}, Assign={list(best.assignment)}")
print("\n" + "="*80)
