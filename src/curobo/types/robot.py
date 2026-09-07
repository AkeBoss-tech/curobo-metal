"""Legacy robot-state and robot-configuration imports."""

from curobo._src.state.state_joint import JointState
from curobo._src.types.robot import RobotCfg

RobotConfig = RobotCfg
__all__ = ["JointState", "RobotCfg", "RobotConfig"]
