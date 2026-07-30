"""Pinned seeded-IK configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


@dataclass
class SeedIKSolverCfg:
    robot_config: RobotCfg
    device_cfg: DeviceCfg = DeviceCfg()
    max_iterations: int = 16
    inner_iterations: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    convergence_position_tolerance: float = 1e-5
    convergence_orientation_tolerance: float = 1e-5
    convergence_joint_limit_weight: float = 1.0
    lambda_initial: float = 0.2
    lambda_factor: float = 2.0
    lambda_max: float = 1e10
    lambda_min: float = 1e-5
    joint_limit_margin: float = 0.001
    batch_success_threshold: float = 1.0
    max_step_size: float = 0.0
    num_seeds: int = 1
    joint_limit_weight: float = 1.0
    use_cuda_graph: bool = True
    use_backward: bool = True
    rho_min: float = 0.001
    tile_threads: int = 32
    sampler_seed: int = 451
    max_problems_mini_batch: int = 200 * 512
    start_cspace_dist_weight: float = 0.01
    position_weight: float = 1.0
    orientation_weight: float = 1.0
    velocity_weight: float = 0.0
    acceleration_weight: float = 0.0

    @staticmethod
    def create(
        robot: Union[str, Dict, RobotCfg],
        device_cfg: DeviceCfg = DeviceCfg(), **kwargs,
    ):
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        else:
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        return SeedIKSolverCfg(robot_cfg, device_cfg, **kwargs)

    def __post_init__(self):
        if self.max_iterations <= 0 or self.inner_iterations <= 0:
            raise ValueError("seed IK iteration counts must be positive")


__all__ = ["SeedIKSolverCfg"]
