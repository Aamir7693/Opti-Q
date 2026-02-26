"""Metrics data structures for LLM DAG Optimizer."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class LLMMetrics:
    """Metrics for LLM execution.

    Attributes:
        final_cost: Total cost (dollars per query)
        final_latency: Total latency (milliseconds)
        final_energy: Total energy (millijoules)
        quality_of_answer: Quality of answer (0-1)
        input_cost: Cost per input token (optional)
        output_cost: Cost per output token (optional)
        input_latency: Latency per input token (optional)
        output_latency: Latency per output token (optional)
        input_energy: Energy per input token (optional)
        output_energy: Energy per output token (optional)
        average_output_tokens: Average output tokens (optional)
    """
    final_cost: float = 0.0
    final_latency: float = 0.0
    final_energy: float = 0.0
    quality_of_answer: float = 0.0

    # Per-token metrics (for detailed calculations)
    input_cost: float = 0.0
    output_cost: float = 0.0
    input_latency: float = 0.0
    output_latency: float = 0.0
    input_energy: float = 0.0
    output_energy: float = 0.0
    average_output_tokens: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple of (cost, latency, energy, qoa).

        Returns:
            Tuple of metrics
        """
        return (self.final_cost, self.final_latency, self.final_energy, self.quality_of_answer)

    @classmethod
    def from_tuple(cls, metrics: Tuple[float, float, float, float]) -> 'LLMMetrics':
        """Create from tuple of (cost, latency, energy, qoa).

        Args:
            metrics: Tuple of (cost, latency, energy, qoa)

        Returns:
            LLMMetrics object
        """
        return cls(
            final_cost=metrics[0],
            final_latency=metrics[1],
            final_energy=metrics[2],
            quality_of_answer=metrics[3]
        )


@dataclass
class Assignment:
    """Assignment of LLMs to DAG nodes with evaluated metrics.

    Attributes:
        node_llms: Tuple of LLM model indices per node (in canonical order)
        metrics: Tuple of (cost, latency, energy, qoa)
    """
    node_llms: Tuple[int, ...]
    metrics: Tuple[float, float, float, float]

    def __hash__(self):
        """Hash for set/dict operations."""
        return hash(self.node_llms)

    @property
    def cost(self) -> float:
        """Get cost from metrics."""
        return self.metrics[0]

    @property
    def latency(self) -> float:
        """Get latency from metrics."""
        return self.metrics[1]

    @property
    def energy(self) -> float:
        """Get energy from metrics."""
        return self.metrics[2]

    @property
    def qoa(self) -> float:
        """Get QoA from metrics."""
        return self.metrics[3]


def convert_metrics_to_dict(metrics: Tuple[float, float, float, float]) -> dict:
    """Convert metrics tuple to dictionary format.

    Args:
        metrics: Tuple of (cost, latency, energy, qoa)

    Returns:
        Dictionary with metrics
    """
    return {
        'final_cost': metrics[0],
        'final_latency': metrics[1],
        'final_energy': metrics[2],
        'quality_of_answer': metrics[3]
    }
