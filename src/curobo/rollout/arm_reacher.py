"""Established reacher-rollout names backed by the portable robot rollout."""

from curobo._src.rollout.rollout_robot import RobotRollout as ArmReacher
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg as ArmReacherConfig

__all__ = ["ArmReacher", "ArmReacherConfig"]
