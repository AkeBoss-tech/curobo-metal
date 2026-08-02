"""Portable configuration object for the pinned MPC solver API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass
class MPCSolverCfg:
    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 1
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.001
    self_collision_check: bool = True
    interpolation_steps: int = 4
    optimization_dt: float = 0.02
    warm_start_optimization_num_iters: int = 200
    cold_start_optimization_num_iters: int = 300
    use_deceleration_on_failure: bool = True
    deceleration_time: Optional[float] = None
    deceleration_profile: str = "exponential"
    max_deceleration_time: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.core_cfg, SolverCoreCfg):
            raise TypeError("core_cfg must be SolverCoreCfg")
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be RobotCfg")
        if self.core_cfg.robot_config is not self.robot_config:
            raise ValueError("core_cfg.robot_config must be robot_config")
        for name in ("max_batch_size", "max_goalset", "num_seeds", "interpolation_steps", "warm_start_optimization_num_iters", "cold_start_optimization_num_iters"):
            _positive_int(name, getattr(self, name))
        if self.interpolation_steps != 4:
            raise ValueError("interpolation_steps must be 4 for MPC")
        for name in ("position_tolerance", "orientation_tolerance", "optimizer_collision_activation_distance", "optimization_dt", "max_deceleration_time"):
            _positive_finite(name, getattr(self, name))
        if self.deceleration_time is not None:
            _positive_finite("deceleration_time", self.deceleration_time)
        if self.deceleration_profile != "exponential":
            raise ValueError("deceleration_profile must be 'exponential'")
        if not isinstance(self.multi_env, bool) or not isinstance(self.self_collision_check, bool) or not isinstance(self.use_deceleration_on_failure, bool):
            raise TypeError("MPC boolean controls must be bool")

    @property
    def device_cfg(self) -> DeviceCfg: return self.core_cfg.device_cfg
    @property
    def use_cuda_graph(self) -> bool: return self.core_cfg.use_cuda_graph
    @property
    def requested_use_cuda_graph(self) -> bool: return self.core_cfg.requested_use_cuda_graph
    @property
    def random_seed(self) -> int: return self.core_cfg.random_seed
    @property
    def store_debug(self) -> bool: return self.core_cfg.store_debug
    @property
    def scene_collision_cfg(self): return self.core_cfg.scene_collision_cfg
    @property
    def optimizer_configs(self): return self.core_cfg.optimizer_configs
    @property
    def optimizer_rollout_configs(self): return self.core_cfg.optimizer_rollout_configs
    @property
    def metrics_rollout_config(self): return self.core_cfg.metrics_rollout_config

    def clone(self, **updates: Any) -> "MPCSolverCfg":
        known = {item.name for item in fields(self)}
        unknown = sorted(set(updates).difference(known))
        if unknown:
            raise TypeError(f"unknown MPCSolverCfg fields: {unknown}")
        core = updates.pop("core_cfg", deepcopy(self.core_cfg))
        robot = updates.pop("robot_config", core.robot_config)
        if robot is not core.robot_config:
            core = replace(core, robot_config=robot)
        return replace(self, core_cfg=core, robot_config=robot, **updates)

    copy = clone

    def update(self, **updates: Any) -> "MPCSolverCfg":
        candidate = self.clone(**updates)
        for item in fields(self):
            setattr(self, item.name, getattr(candidate, item.name))
        return self

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["mpc/lbfgs_mpc.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "mpc/transition_bspline_mpc.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None},
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        optimization_dt: float = 0.02,
        interpolation_steps: int = 4,
        use_deceleration_on_failure: bool = True,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
        max_deceleration_time: float = 2.0,
        load_collision_spheres: bool = True,
        num_control_points: Optional[int] = None,
        squared_l2_regularization_weight: Optional[List[float]] = None,
        warm_start_optimization_num_iters: int = 200,
        cold_start_optimization_num_iters: int = 300,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        **kwargs: Any,
    ) -> "MPCSolverCfg":
        if kwargs:
            raise TypeError(f"unsupported MPC configuration fields: {sorted(kwargs)}")
        if interpolation_steps != 4:
            raise ValueError("interpolation_steps must be 4 for MPC")
        del metrics_rollout, transition_model, collision_cache
        del override_optimizer_num_iters, transition_model_config_instance_type, cost_manager_config_instance_type
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        elif isinstance(robot, str):
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg,
                                                      load_collision_spheres=load_collision_spheres)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        else:
            robot_cfg = RobotCfg.create(robot, device_cfg=device_cfg,
                                        load_collision_spheres=load_collision_spheres)
        if num_control_points is not None:
            _positive_int("num_control_points", num_control_points)
        optimizer_records = deepcopy(list(optimizer_configs))
        if squared_l2_regularization_weight is not None:
            if len(squared_l2_regularization_weight) == 0:
                raise ValueError("squared_l2_regularization_weight must not be empty")
            for value in squared_l2_regularization_weight:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("squared_l2_regularization_weight values must be finite")
        core = SolverCoreCfg(robot_cfg, device_cfg, optimizer_records,
                             scene_collision_cfg=scene_model, use_cuda_graph=use_cuda_graph,
                             random_seed=random_seed, store_debug=store_debug)
        return MPCSolverCfg(
            core, robot_cfg, max_batch_size, multi_env, max_goalset,
            position_tolerance=position_tolerance, orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            self_collision_check=self_collision_check, interpolation_steps=interpolation_steps,
            optimization_dt=optimization_dt, warm_start_optimization_num_iters=warm_start_optimization_num_iters,
            cold_start_optimization_num_iters=cold_start_optimization_num_iters,
            use_deceleration_on_failure=use_deceleration_on_failure, deceleration_time=deceleration_time,
            deceleration_profile=deceleration_profile, max_deceleration_time=max_deceleration_time,
        )


__all__ = ["MPCSolverCfg"]
