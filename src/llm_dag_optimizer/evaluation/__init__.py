"""
Evaluation module - imports from main.py for now.

This is a transitional module that imports evaluation functions from main.py.
Over time, these can be migrated here for full modularization.
"""

import sys
from pathlib import Path

# Add parent directory to import main.py
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import evaluation functions from main.py
try:
    from old.main import (
        estimate_schedule_v3 as evaluate_dag_plan,
        get_single_model_metrics,
        get_pairwise_metrics,
        match_subDAGs_v3 as match_subdags,
        new_traversal_v3 as traverse_for_qoa,
        new_traversal_just_for_cost_latency_energy_v3 as traverse_for_cost,
        calculate_cost_parallel as compute_parallel_cost,
        calculate_cost_sequential_v2 as compute_sequential_cost,
        LLMMetrics,
        BFSNode,
        ProcessingTableEntry,
        BlendingNodeDBReference,
    )
except ImportError as e:
    print(f"Warning: Could not import from main.py: {e}")
    # Provide stub implementations
    def evaluate_dag_plan(*args, **kwargs):
        raise NotImplementedError("Evaluation logic not available")

__all__ = [
    "evaluate_dag_plan",
    "get_single_model_metrics",
    "get_pairwise_metrics",
    "match_subdags",
    "traverse_for_qoa",
    "traverse_for_cost",
    "compute_parallel_cost",
    "compute_sequential_cost",
    "LLMMetrics",
    "BFSNode",
    "ProcessingTableEntry",
    "BlendingNodeDBReference",
]
