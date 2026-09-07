"""Established rollout-metrics import path."""

from curobo._src.rollout.metrics import RolloutMetrics

CuroboMetrics = RolloutMetrics
CuroboGroupMetrics = RolloutMetrics
__all__ = ["CuroboGroupMetrics", "CuroboMetrics"]
