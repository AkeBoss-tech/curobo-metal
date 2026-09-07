"""Established rollout base names backed by the portable rollout core."""

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.rollout.rollout_robot import RobotRollout

Goal = GoalRegistry
RolloutBase = RobotRollout
__all__ = ["Goal", "RolloutBase", "RolloutMetrics"]
