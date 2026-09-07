"""Established arm-rollout names backed by the portable robot rollout."""

from curobo._src.rollout.rollout_robot import RobotRollout as ArmBase
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg as ArmBaseConfig

__all__ = ["ArmBase", "ArmBaseConfig"]
