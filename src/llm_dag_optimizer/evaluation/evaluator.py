"""
Main evaluation logic for DAG plans.

This module contains the core evaluation system that estimates metrics
(cost, latency, energy, QoA) for LLM execution plans.
"""

import copy
import collections
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from ..structures.metrics import LLMMetrics


@dataclass
class BFSNode:
    """Node in BFS traversal with accumulated metrics."""
    LLM_name: int
    node: int
    metrics: LLMMetrics
    accumulated_metrics: LLMMetrics
    first_node_in_seq: bool


@dataclass
class ProcessingTableEntry:
    """Entry in processing table for parallel nodes."""
    node: int
    metrics: LLMMetrics
    accumulated_metrics: LLMMetrics


@dataclass
class BlendingNodeDBReference:
    """Reference to historical blending node data."""
    inputs: Dict[int, LLMMetrics]
    output: float  # QoA


def evaluate_dag_plan(
    G: nx.DiGraph,
    assignment: List[int],
    query_type: str,
    df_history: pd.DataFrame,
    query_tokens: int = 215,
    blending_prompt_tokens: int = 26,
    ctx_tokens: int = 39,
    turn_off_exact_fuzzy_matching: bool = True
) -> Tuple[float, float, float, float]:
    """
    Main evaluation function for DAG plans.

    This is the primary entry point for evaluating a DAG with LLM assignments.
    It uses historical performance data to estimate metrics.

    Args:
        G: DAG structure as NetworkX DiGraph
        assignment: List of LLM model indices per node
        query_type: Query category (e.g., "Art", "Science and technology")
        df_history: Historical performance DataFrame
        query_tokens: Number of query tokens
        blending_prompt_tokens: Blending prompt token count
        ctx_tokens: Context token count
        turn_off_exact_fuzzy_matching: If True, use exact matching only

    Returns:
        Tuple of (cost, latency, energy, qoa)
    """
    # Import here to avoid circular dependency
    # This function will be implemented by extracting from main.py
    raise NotImplementedError("To be extracted from main.py estimate_schedule_v3()")


# Placeholder - will extract these from main.py
def get_single_model_metrics(
    llm_assignment: List[str],
    query_type: str,
    df_history: pd.DataFrame
) -> Dict[int, LLMMetrics]:
    """Get single model baseline metrics."""
    raise NotImplementedError("To be extracted from main.py")


def get_pairwise_metrics(
    llm_assignment: List[str],
    query_type: str,
    df_history: pd.DataFrame
) -> Dict[Tuple[int, int], LLMMetrics]:
    """Get pairwise sequential execution metrics."""
    raise NotImplementedError("To be extracted from main.py")
