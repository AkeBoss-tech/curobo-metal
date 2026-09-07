"""Legacy names for the portable kinematics implementation."""

from curobo._src.robot.kinematics.kinematics import Kinematics as CudaRobotModel
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg as CudaRobotModelConfig

__all__ = ["CudaRobotModel", "CudaRobotModelConfig"]
