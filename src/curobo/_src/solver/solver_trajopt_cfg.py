"""Portable trajectory-optimization configuration at cuRobo's pinned path."""

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
from curobo._src.util.trajectory import TrajInterpolationType


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass
class TrajOptSolverCfg:
    """Runtime-ready TrajOpt options backed by portable rollout records.

    The additional ``action_horizon``, ``max_iterations`` and
    ``optimizer_name`` controls describe the composed PyTorch trajectory
    solver.  They are intentionally separate from CUDA-only B-spline graph
    internals, so applications can change the dependency name without
    receiving a falsely equivalent CUDA graph object.
    """

    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.0
    self_collision_check: bool = True
    minimum_trajectory_dt: float = 0.002
    maximum_trajectory_dt: float = 0.2
    interpolation_dt: float = 0.025
    interpolation_type: TrajInterpolationType = TrajInterpolationType.BSPLINE_KNOTS_CUDA
    interpolation_buffer_size: int = 1000
    action_horizon: int = 32
    max_iterations: int = 200
    optimizer_name: str = "adam"
    # Kept for source compatibility with early portable direct constructors.
    # Runtime accessors remain sourced from the coherent SolverCoreCfg.
    use_cuda_graph_value: bool = False
    random_seed_value: int = 123
    store_debug_value: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be RobotCfg")
        # Earlier portable releases documented ``core_cfg=[]`` as a compact
        # direct-construction form.  Continue accepting it, but normalize it
        # immediately to the full core value object used by the factory.
        if not isinstance(self.core_cfg, SolverCoreCfg):
            if not isinstance(self.core_cfg, list):
                raise TypeError("core_cfg must be SolverCoreCfg or a list of optimizer configs")
            self.core_cfg = SolverCoreCfg(
                self.robot_config, self.robot_config.device_cfg,
                optimizer_configs=list(self.core_cfg), use_cuda_graph=self.use_cuda_graph_value,
                random_seed=self.random_seed_value, store_debug=self.store_debug_value,
            )
        if self.core_cfg.robot_config is not self.robot_config:
            raise ValueError("core_cfg.robot_config must be robot_config")
        for name in ("max_batch_size", "max_goalset", "num_seeds", "interpolation_buffer_size", "action_horizon", "max_iterations"):
            _positive_int(name, getattr(self, name))
        for name in (
            "position_tolerance", "orientation_tolerance", "optimizer_collision_activation_distance",
            "minimum_trajectory_dt", "maximum_trajectory_dt", "interpolation_dt",
        ):
            _positive_finite(name, getattr(self, name))
        if self.minimum_trajectory_dt > self.maximum_trajectory_dt:
            raise ValueError("minimum_trajectory_dt must not exceed maximum_trajectory_dt")
        if not isinstance(self.interpolation_type, TrajInterpolationType):
            raise TypeError("interpolation_type must be TrajInterpolationType")
        if self.interpolation_type in (TrajInterpolationType.QUARTIC, TrajInterpolationType.BSPLINE_KNOTS_CUDA):
            # BSPLINE_KNOTS_CUDA is retained as the upstream default and maps
            # to the portable cubic trajectory implementation at execution.
            pass
        if self.optimizer_name not in {"adam", "lbfgs", "particle", "es"}:
            raise ValueError("optimizer_name must be 'adam', 'lbfgs', 'particle', or 'es'")
        if not isinstance(self.multi_env, bool) or not isinstance(self.self_collision_check, bool):
            raise TypeError("TrajOpt boolean controls must be bool")

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

    def clone(self, **updates: Any) -> "TrajOptSolverCfg":
        known = {item.name for item in fields(self)}
        unknown = sorted(set(updates).difference(known))
        if unknown:
            raise TypeError(f"unknown TrajOptSolverCfg fields: {unknown}")
        core = updates.pop("core_cfg", deepcopy(self.core_cfg))
        robot = updates.pop("robot_config", core.robot_config)
        if robot is not core.robot_config:
            core = replace(core, robot_config=robot)
        return replace(self, core_cfg=core, robot_config=robot, **updates)

    copy = clone

    def update(self, **updates: Any) -> "TrajOptSolverCfg":
        candidate = self.clone(**updates)
        for item in fields(self):
            setattr(self, item.name, getattr(candidate, item.name))
        return self

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 4,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        minimum_trajectory_dt: float = 0.002,
        maximum_trajectory_dt: float = 0.2,
        load_collision_spheres: bool = True,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None},
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: int = 1000,
    ) -> "TrajOptSolverCfg":
        _positive_finite("interpolation_dt", interpolation_dt)
        _positive_int("interpolation_buffer_size", interpolation_buffer_size)
        num_envs = max_batch_size if multi_env else 1
        del metrics_rollout, transition_model, collision_cache
        del transition_model_config_instance_type, cost_manager_config_instance_type
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        elif isinstance(robot, str):
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg,
                                                      load_collision_spheres=load_collision_spheres)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        else:
            robot_cfg = RobotCfg.create(robot, device_cfg=device_cfg,
                                        load_collision_spheres=load_collision_spheres,
                                        num_envs=num_envs)
        interpolation_type = TrajInterpolationType.BSPLINE_KNOTS_CUDA
        core = SolverCoreCfg(robot_cfg, device_cfg, deepcopy(list(optimizer_configs)),
                             scene_collision_cfg=scene_model, use_cuda_graph=use_cuda_graph,
                             random_seed=random_seed, store_debug=store_debug)
        max_iterations = 200
        for name in ("lbfgs", "particle", "adam", "es"):
            override = (override_optimizer_num_iters or {}).get(name)
            if override is not None:
                _positive_int(f"override_optimizer_num_iters[{name!r}]", override)
                max_iterations = override
                break
        names = " ".join(map(str, optimizer_configs)).lower()
        optimizer_name = "lbfgs" if "lbfgs" in names else ("particle" if "particle" in names else "adam")
        return TrajOptSolverCfg(
            core, robot_cfg, max_batch_size, multi_env, max_goalset, num_seeds,
            position_tolerance, orientation_tolerance, optimizer_collision_activation_distance,
            0.0, self_collision_check, minimum_trajectory_dt, maximum_trajectory_dt,
            interpolation_dt, interpolation_type, interpolation_buffer_size,
            max_iterations=max_iterations, optimizer_name=optimizer_name,
            use_cuda_graph_value=use_cuda_graph, random_seed_value=random_seed,
            store_debug_value=store_debug,
        )


__all__ = ["TrajOptSolverCfg"]
