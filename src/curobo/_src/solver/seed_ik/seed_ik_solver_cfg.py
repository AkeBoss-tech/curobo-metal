"""Pinned seeded-IK configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Mapping, Union

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
        robot: Union[str, PathLike[str], Mapping[str, Any], RobotCfg],
        device_cfg: DeviceCfg = DeviceCfg(), **kwargs,
    ) -> "SeedIKSolverCfg":
        """Create a solver configuration from any portable robot source.

        Mappings and externally-addressable paths use ``RobotCfg.create``.
        A bare packaged cuRobo config name (for example ``"franka.yml"``)
        retains the upstream content-path lookup through ``KinematicsCfg``.
        In both cases the solver owns a public ``RobotCfg``, rather than a
        kinematics-only mapping.
        """
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        elif isinstance(robot, Mapping):
            robot_cfg = RobotCfg.create(dict(robot), device_cfg=device_cfg)
        elif isinstance(robot, PathLike) or (isinstance(robot, str) and Path(robot).exists()):
            robot_cfg = RobotCfg.create(robot, device_cfg=device_cfg)
        elif isinstance(robot, str):
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        else:
            raise TypeError("robot must be a path, mapping, or RobotCfg")
        return SeedIKSolverCfg(robot_config=robot_cfg, device_cfg=device_cfg, **kwargs)

    def __post_init__(self):
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be a RobotCfg")
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if self.max_iterations <= 0 or self.inner_iterations <= 0:
            raise ValueError("seed IK iteration counts must be positive")
        if self.max_iterations < self.inner_iterations:
            raise ValueError("max_iterations must be >= inner_iterations")
        if self.max_iterations % self.inner_iterations:
            raise ValueError("max_iterations must be divisible by inner_iterations")
        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be positive")
        if self.max_problems_mini_batch <= 0:
            raise ValueError("max_problems_mini_batch must be positive")
        if self.tile_threads <= 0:
            raise ValueError("tile_threads must be positive")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int):
            raise TypeError("sampler_seed must be an integer")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be non-negative")

        numeric = {
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "convergence_position_tolerance": self.convergence_position_tolerance,
            "convergence_orientation_tolerance": self.convergence_orientation_tolerance,
            "convergence_joint_limit_weight": self.convergence_joint_limit_weight,
            "lambda_initial": self.lambda_initial,
            "lambda_factor": self.lambda_factor,
            "lambda_max": self.lambda_max,
            "lambda_min": self.lambda_min,
            "joint_limit_margin": self.joint_limit_margin,
            "batch_success_threshold": self.batch_success_threshold,
            "max_step_size": self.max_step_size,
            "joint_limit_weight": self.joint_limit_weight,
            "rho_min": self.rho_min,
            "start_cspace_dist_weight": self.start_cspace_dist_weight,
            "position_weight": self.position_weight,
            "orientation_weight": self.orientation_weight,
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if min(
            self.position_tolerance, self.orientation_tolerance,
            self.convergence_position_tolerance, self.convergence_orientation_tolerance,
            self.convergence_joint_limit_weight, self.joint_limit_weight,
            self.rho_min, self.max_step_size, self.start_cspace_dist_weight,
            self.position_weight, self.orientation_weight, self.velocity_weight,
            self.acceleration_weight,
        ) < 0:
            raise ValueError("seed IK tolerances and weights must be non-negative")
        if not 0.0 <= self.joint_limit_margin < 0.5:
            raise ValueError("joint_limit_margin must be in [0, 0.5)")
        if self.lambda_min <= 0 or self.lambda_max < self.lambda_min:
            raise ValueError("lambda_min must be positive and lambda_max must be >= lambda_min")
        if not self.lambda_min <= self.lambda_initial <= self.lambda_max:
            raise ValueError("lambda_initial must be in [lambda_min, lambda_max]")
        if self.lambda_factor <= 1:
            raise ValueError("lambda_factor must be greater than one")
        if not 0.0 <= self.batch_success_threshold <= 1.0:
            raise ValueError("batch_success_threshold must be in [0, 1]")
        if not isinstance(self.use_cuda_graph, bool) or not isinstance(self.use_backward, bool):
            raise TypeError("use_cuda_graph and use_backward must be bool")


__all__ = ["SeedIKSolverCfg"]
