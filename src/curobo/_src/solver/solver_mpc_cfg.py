"""Portable configuration object for the pinned MPC solver API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg, create_scene_collision_cfg, resolve_yaml_configs
from curobo._src.util.logging import log_and_raise
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
        for name in ("position_tolerance", "orientation_tolerance", "optimizer_collision_activation_distance", "optimization_dt", "max_deceleration_time"):
            _positive_finite(name, getattr(self, name))
        if (isinstance(self.non_terminal_tool_pose_weight_factor, bool)
                or not isinstance(self.non_terminal_tool_pose_weight_factor, (float, int))
                or not math.isfinite(self.non_terminal_tool_pose_weight_factor)
                or self.non_terminal_tool_pose_weight_factor < 0):
            raise ValueError(
                "non_terminal_tool_pose_weight_factor must be finite and nonnegative"
            )
        if self.deceleration_time is not None:
            _positive_finite("deceleration_time", self.deceleration_time)
        if self.deceleration_profile not in {"exponential", "linear"}:
            raise ValueError("deceleration_profile must be 'exponential' or 'linear'")
        if self.interpolation_steps != 4:
            raise ValueError("Interpolation steps must be 4 for MPC")
        if not isinstance(self.multi_env, bool) or not isinstance(self.self_collision_check, bool) or not isinstance(self.use_deceleration_on_failure, bool):
            raise TypeError("MPC boolean controls must be bool")

    @property
    def device_cfg(self) -> DeviceCfg: return self.core_cfg.device_cfg
    @property
    def use_cuda_graph(self) -> bool: return self.core_cfg.use_cuda_graph
    @property
    def _requested_use_cuda_graph_portable(self) -> bool: return self.core_cfg.requested_use_cuda_graph
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

    def _clone_portable(self, **updates: Any) -> "MPCSolverCfg":
        known = {item.name for item in fields(self)}
        unknown = sorted(set(updates).difference(known))
        if unknown:
            raise TypeError(f"unknown MPCSolverCfg fields: {unknown}")
        core = updates.pop("core_cfg", deepcopy(self.core_cfg))
        robot = updates.pop("robot_config", core.robot_config)
        if robot is not core.robot_config:
            core = replace(core, robot_config=robot)
        return replace(self, core_cfg=core, robot_config=robot, **updates)

    copy = _clone_portable

    def _update_portable(self, **updates: Any) -> "MPCSolverCfg":
        candidate = self._clone_portable(**updates)
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
        **kwargs,
    ) -> MPCSolverCfg:
        if kwargs:
            raise TypeError(f"unsupported MPC configuration fields: {sorted(kwargs)}")
        robot_cfg, optimizer_dicts, metrics_dict, transition_dict, scene_dict = resolve_yaml_configs(
            robot, optimizer_configs, metrics_rollout, transition_model, scene_model,
            device_cfg, load_collision_spheres, max_batch_size if multi_env else 1,
        )
        if num_control_points is not None:
            _positive_int("num_control_points", num_control_points)
        optimizer_records = optimizer_dicts
        if squared_l2_regularization_weight is not None:
            if len(squared_l2_regularization_weight) == 0:
                raise ValueError("squared_l2_regularization_weight must not be empty")
            for value in squared_l2_regularization_weight:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("squared_l2_regularization_weight values must be finite")
        core = SolverCoreCfg(
            robot_cfg, device_cfg, optimizer_records,
            optimizer_rollout_configs=[deepcopy(transition_dict) for _ in optimizer_records],
            metrics_rollout_config=deepcopy(metrics_dict),
            scene_collision_cfg=create_scene_collision_cfg(scene_dict, collision_cache, device_cfg),
            use_cuda_graph=use_cuda_graph, random_seed=random_seed, store_debug=store_debug,
        )
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

# Retain portable clone/update conveniences without including them in the
# pinned class declaration surface.
MPCSolverCfg.clone = MPCSolverCfg._clone_portable
MPCSolverCfg.update = MPCSolverCfg._update_portable
MPCSolverCfg.requested_use_cuda_graph = MPCSolverCfg._requested_use_cuda_graph_portable


__all__ = ["MPCSolverCfg"]
