"""
Mutation operators for genetic algorithms.

This module provides mutation operators used by NSGA-II and other
evolutionary algorithms.
"""

from ..structures.individual import mutate as mutate_individual

__all__ = ["mutate_individual"]
