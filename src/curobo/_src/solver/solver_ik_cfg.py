"""Portable configuration value object for the pinned IK solver API.

The upstream factory assembles CUDA/Warp rollout graphs from several YAML
files.  This module keeps that input contract but assembles the ordinary
PyTorch CPU/MPS rollout records used by :mod:`curobo._src.solver.solver_ik`.
CUDA graph capture remains a deliberately explicit backend boundary: the
request is retained on ``core_cfg.requested_use_cuda_graph`` while the active
portable configuration always reports ``use_cuda_graph == False``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
import math
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_core_cfg import (
    SolverCoreCfg, create_scene_collision_cfg, create_solver_core_cfg,
    resolve_yaml_configs,
)
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
class _IKSolverCfgPortableMixin:
    """Configuration specific to portable batched inverse kinematics."""

    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 32
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.0
    success_requires_convergence: bool = True
    override_iters_for_multi_link_ik: Optional[int] = None
    use_lm_seed: bool = True
    exit_early: bool = True
    exit_early_batch_success_threshold: float = 1.0
    optimization_dt: Optional[float] = None
    seed_position_weight: float = 1.0
    seed_orientation_weight: float = 1.0
    seed_velocity_weight: float = 0.0
    seed_acceleration_weight: float = 0.0
    seed_solver_num_seeds: int = 32
    self_collision_check: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.core_cfg, SolverCoreCfg):
            raise TypeError("core_cfg must be SolverCoreCfg")
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be RobotCfg")
        if self.core_cfg.robot_config is not self.robot_config:
            raise ValueError("core_cfg.robot_config must be robot_config")
        for name in ("max_batch_size", "max_goalset", "num_seeds", "seed_solver_num_seeds"):
            _positive_int(name, getattr(self, name))
        if self.override_iters_for_multi_link_ik is not None:
            _positive_int("override_iters_for_multi_link_ik", self.override_iters_for_multi_link_ik)
        for name in (
            "position_tolerance", "orientation_tolerance",
            "optimizer_collision_activation_distance", "seed_position_weight",
            "seed_orientation_weight",
        ):
            _positive_finite(name, getattr(self, name))
        for name in (
            "non_terminal_tool_pose_weight_factor", "seed_velocity_weight",
            "seed_acceleration_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.optimization_dt is not None:
            _positive_finite("optimization_dt", self.optimization_dt)
        if not isinstance(self.exit_early_batch_success_threshold, (float, int)) or not math.isfinite(self.exit_early_batch_success_threshold) or not 0 < self.exit_early_batch_success_threshold <= 1:
            raise ValueError("exit_early_batch_success_threshold must be in (0, 1]")
        for name in ("multi_env", "success_requires_convergence", "use_lm_seed", "exit_early", "self_collision_check"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

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

    def clone(self, **updates: Any) -> "IKSolverCfg":
        """Return an independent configuration, optionally with validated updates."""
        known = {item.name for item in fields(self)}
        unknown = sorted(set(updates).difference(known))
        if unknown:
            raise TypeError(f"unknown IKSolverCfg fields: {unknown}")
        core = updates.pop("core_cfg", deepcopy(self.core_cfg))
        robot = updates.pop("robot_config", core.robot_config)
        if robot is not core.robot_config:
            # A caller who intentionally changes robot configuration gets a
            # coherent new core record instead of a silently split config.
            core = replace(core, robot_config=robot)
        return replace(self, core_cfg=core, robot_config=robot, **updates)

    copy = clone

    def update(self, **updates: Any) -> "IKSolverCfg":
        """Mutate this value object only after validating the complete update."""
        candidate = self.clone(**updates)
        for item in fields(self):
            setattr(self, item.name, getattr(candidate, item.name))
        return self

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 32,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"particle": None, "lbfgs": None},
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        override_iters_for_multi_link_ik: Optional[int] = None,
        optimization_dt: Optional[float] = None,
        load_collision_spheres: bool = True,
        velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None,
        success_requires_convergence: bool = True,
        seed_position_weight: float = 1.0,
        seed_orientation_weight: float = 1.0,
        seed_velocity_weight: float = 0.0,
        seed_acceleration_weight: float = 0.0,
        seed_solver_num_seeds: int = 32,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
    ) -> "IKSolverCfg":
        """Compile flexible robot/YAML inputs into CPU/MPS solver records.

        ``use_cuda_graph=True`` is accepted for dependency-name compatibility;
        portable execution uses a persistent eager cache instead of emulating
        NVIDIA CUDA Graph capture.
        """
        robot_cfg, optimizer_dicts, metrics_dict, transition_dict, scene_dict = resolve_yaml_configs(
            robot, optimizer_configs, metrics_rollout, transition_model, scene_model,
            device_cfg, load_collision_spheres,
            max_batch_size if multi_env else 1,
        )
        # Task YAML is an optional upstream content bundle.  Keep the supplied
        # records for inspection without requiring CUDA-only task assets, but
        # do honour the two documented regularization overrides when a caller
        # provides a structured optimizer record.
        optimizer_records = optimizer_dicts
        for label, value in (("velocity_regularization_weight", velocity_regularization_weight),
                             ("acceleration_regularization_weight", acceleration_regularization_weight)):
            if value is not None:
                _positive_finite(label, value)
        if velocity_regularization_weight is not None or acceleration_regularization_weight is not None:
            for record in optimizer_records:
                if not isinstance(record, dict):
                    continue
                rollout = record.setdefault("rollout", {})
                if not isinstance(rollout, dict):
                    raise TypeError("optimizer rollout configuration must be a mapping")
                cost_cfg = rollout.setdefault("cost_cfg", {})
                if not isinstance(cost_cfg, dict):
                    raise TypeError("optimizer rollout cost_cfg must be a mapping")
                cspace_cfg = cost_cfg.setdefault("cspace_cfg", {})
                if not isinstance(cspace_cfg, dict):
                    raise TypeError("optimizer rollout cspace_cfg must be a mapping")
                values = list(cspace_cfg.get("squared_l2_regularization_weight", [0.0, 0.0]))
                if len(values) != 2:
                    raise ValueError(
                        "squared_l2_regularization_weight must contain velocity and acceleration entries"
                    )
                if velocity_regularization_weight is not None:
                    values[0] = velocity_regularization_weight
                if acceleration_regularization_weight is not None:
                    values[1] = acceleration_regularization_weight
                cspace_cfg["squared_l2_regularization_weight"] = values
        core = SolverCoreCfg(
            robot_cfg, device_cfg, optimizer_records,
            optimizer_rollout_configs=[deepcopy(transition_dict) for _ in optimizer_records],
            metrics_rollout_config=deepcopy(metrics_dict),
            scene_collision_cfg=create_scene_collision_cfg(scene_dict, collision_cache, device_cfg),
            use_cuda_graph=use_cuda_graph, random_seed=random_seed, store_debug=store_debug,
        )
        return IKSolverCfg(
            core, robot_cfg, max_batch_size, multi_env, max_goalset, num_seeds,
            position_tolerance, orientation_tolerance, optimizer_collision_activation_distance,
            success_requires_convergence=success_requires_convergence,
            override_iters_for_multi_link_ik=override_iters_for_multi_link_ik,
            optimization_dt=optimization_dt, seed_position_weight=seed_position_weight,
            seed_orientation_weight=seed_orientation_weight, seed_velocity_weight=seed_velocity_weight,
            seed_acceleration_weight=seed_acceleration_weight, seed_solver_num_seeds=seed_solver_num_seeds,
            self_collision_check=self_collision_check,
        )


@dataclass
class IKSolverCfg(_IKSolverCfgPortableMixin):
    """Pinned IK configuration declaration backed by portable construction."""

    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 32
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.0
    success_requires_convergence: bool = True
    override_iters_for_multi_link_ik: Optional[int] = None
    use_lm_seed: bool = True
    exit_early: bool = True
    exit_early_batch_success_threshold: float = 1.0
    optimization_dt: Optional[float] = None
    seed_position_weight: float = 1.0
    seed_orientation_weight: float = 1.0
    seed_velocity_weight: float = 0.0
    seed_acceleration_weight: float = 0.0
    seed_solver_num_seeds: int = 32
    self_collision_check: bool = True

    @property
    def device_cfg(self) -> DeviceCfg:
        return self.core_cfg.device_cfg
    @property
    def use_cuda_graph(self) -> bool:
        return self.core_cfg.use_cuda_graph
    @property
    def random_seed(self) -> int:
        return self.core_cfg.random_seed
    @property
    def store_debug(self) -> bool:
        return self.core_cfg.store_debug
    @property
    def scene_collision_cfg(self):
        return self.core_cfg.scene_collision_cfg
    @property
    def optimizer_configs(self):
        return self.core_cfg.optimizer_configs
    @property
    def optimizer_rollout_configs(self):
        return self.core_cfg.optimizer_rollout_configs
    @property
    def metrics_rollout_config(self):
        return self.core_cfg.metrics_rollout_config

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(), num_seeds: int = 32,
        position_tolerance: float = 0.005, orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True, random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01, store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"particle": None, "lbfgs": None},
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        override_iters_for_multi_link_ik: Optional[int] = None, optimization_dt: Optional[float] = None,
        load_collision_spheres: bool = True, velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None, success_requires_convergence: bool = True,
        seed_position_weight: float = 1.0, seed_orientation_weight: float = 1.0,
        seed_velocity_weight: float = 0.0, seed_acceleration_weight: float = 0.0,
        seed_solver_num_seeds: int = 32, max_batch_size: int = 1, multi_env: bool = False,
        max_goalset: int = 1,
    ) -> IKSolverCfg:
        return _IKSolverCfgPortableMixin.create(
            robot, optimizer_configs, metrics_rollout, transition_model, scene_model, collision_cache,
            self_collision_check, device_cfg, num_seeds, position_tolerance, orientation_tolerance,
            use_cuda_graph, random_seed, optimizer_collision_activation_distance, store_debug,
            override_optimizer_num_iters, transition_model_config_instance_type,
            cost_manager_config_instance_type, override_iters_for_multi_link_ik, optimization_dt,
            load_collision_spheres, velocity_regularization_weight, acceleration_regularization_weight,
            success_requires_convergence, seed_position_weight, seed_orientation_weight,
            seed_velocity_weight, seed_acceleration_weight, seed_solver_num_seeds, max_batch_size,
            multi_env, max_goalset,
        )


__all__ = ["IKSolverCfg"]
