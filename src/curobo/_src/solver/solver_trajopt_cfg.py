"""Portable trajectory-optimization configuration at cuRobo's pinned path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import math
import os
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType
from curobo._src.util.config_io import join_path, resolve_config
from curobo.content import get_scene_configs_path


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _copy_optimizer_configs(value: List[Union[str, Dict[str, Any]]]) -> List[Union[str, Dict[str, Any]]]:
    """Copy the YAML-shaped optimizer transport contract without mutating it."""
    if not isinstance(value, (list, tuple)):
        raise TypeError("optimizer_configs must be a list or tuple")
    if not value:
        raise ValueError("optimizer_configs must contain at least one optimizer")
    for index, item in enumerate(value):
        if not isinstance(item, (str, dict)):
            raise TypeError(f"optimizer_configs[{index}] must be a path or mapping")
    return deepcopy(list(value))


def _optimizer_name(records: List[Union[str, Dict[str, Any]]]) -> str:
    """Choose the portable optimizer that represents the first V2 stage.

    V2 permits a YAML pipeline.  The standalone portable TrajOpt solver has a
    single execution stage, so it deliberately chooses the first declared
    stage rather than silently treating every name as L-BFGS.
    """
    first = records[0]
    if isinstance(first, dict):
        payload = first.get("optimizer", first)
        if not isinstance(payload, dict):
            raise TypeError("optimizer configuration 'optimizer' value must be a mapping")
        token = str(payload.get("solver_type", payload.get("type", "lbfgs"))).lower()
    else:
        token = first.lower()
    if any(name in token for name in ("particle", "mppi")):
        return "particle"
    if token in {"es", "evolution", "evolution_strategies"} or "evolution" in token:
        return "es"
    if "lbfgs" in token or "lsr1" in token or "conjugate" in token:
        return "lbfgs"
    if "adam" in token or "gradient" in token:
        return "adam"
    raise ValueError(
        "optimizer_configs first stage is not portable; use an Adam, L-BFGS, particle, or ES optimizer"
    )


def _transition_interpolation_type(value: Union[str, Dict[str, Any]]) -> TrajInterpolationType:
    """Resolve V2's position-control interpolation choice without CUDA kernels."""
    candidate: Any = value
    if isinstance(value, str):
        path = join_path(get_scene_configs_path().parent / "task", value)
        # Task YAML is optional in this minimal distribution.  If an app
        # provides one, use its declared control space; otherwise retain the
        # pinned B-spline default as a portable linear-execution request.
        try:
            candidate = resolve_config(path) if os.path.exists(path) else value
        except (OSError, ValueError):
            candidate = value
    if isinstance(candidate, dict):
        candidate = candidate.get("transition_model_cfg", candidate)
        if not isinstance(candidate, dict):
            raise TypeError("transition_model_cfg must be a mapping")
        control_space = candidate.get("control_space")
        if control_space is None:
            return TrajInterpolationType.BSPLINE_KNOTS_CUDA
        text = getattr(control_space, "name", str(control_space)).upper()
        return (
            TrajInterpolationType.LINEAR_CUDA
            if text == "POSITION" or text.endswith(".POSITION")
            else TrajInterpolationType.BSPLINE_KNOTS_CUDA
        )
    if not isinstance(candidate, str):
        raise TypeError("transition_model must be a path or mapping")
    return TrajInterpolationType.LINEAR_CUDA if "position" in candidate.lower() else TrajInterpolationType.BSPLINE_KNOTS_CUDA


def _scene_cfg(
    scene_model: Optional[Union[str, Dict[str, Any]]], collision_cache: Optional[Dict[str, int]],
    device_cfg: DeviceCfg, num_envs: int,
) -> Optional[SceneCollisionCfg]:
    """Compile scene/cache values into the shared portable collision record."""
    if collision_cache is not None:
        if not isinstance(collision_cache, dict):
            raise TypeError("collision_cache must be a mapping or None")
        for key, capacity in collision_cache.items():
            if not isinstance(key, str):
                raise TypeError("collision_cache keys must be strings")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
                raise ValueError(f"collision_cache[{key!r}] must be a nonnegative integer")
    if scene_model is None and collision_cache is None:
        return None
    if isinstance(scene_model, SceneCollisionCfg):
        if scene_model.device_cfg != device_cfg:
            raise ValueError("scene_model SceneCollisionCfg device_cfg must match device_cfg")
        if collision_cache is not None and scene_model.cache != collision_cache:
            raise ValueError("collision_cache cannot override a prebuilt SceneCollisionCfg cache")
        if scene_model.num_envs != num_envs:
            raise ValueError("scene_model SceneCollisionCfg.num_envs must match configured environments")
        return scene_model
    scene = scene_model
    if isinstance(scene_model, str):
        scene = resolve_config(join_path(get_scene_configs_path(), scene_model))
    if scene is not None and not isinstance(scene, (dict, list)):
        raise TypeError("scene_model must be a scene path, mapping, list, SceneCollisionCfg, or None")
    result = SceneCollisionCfg(device_cfg=device_cfg, scene_model=scene, num_envs=num_envs,
                               cache=deepcopy(collision_cache))
    if result.num_envs != num_envs:
        raise ValueError("scene_model environment count must match configured environments")
    return result


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
        _nonnegative_finite(
            "non_terminal_tool_pose_weight_factor", self.non_terminal_tool_pose_weight_factor
        )
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
        if (not isinstance(self.multi_env, bool) or not isinstance(self.self_collision_check, bool)
                or not isinstance(self.use_cuda_graph_value, bool)
                or not isinstance(self.store_debug_value, bool)):
            raise TypeError("TrajOpt boolean controls must be bool")
        if isinstance(self.random_seed_value, bool) or not isinstance(self.random_seed_value, int):
            raise TypeError("random_seed_value must be an integer")
        if self.random_seed_value < 0:
            raise ValueError("random_seed_value must be nonnegative")

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

    @property
    def portable_interpolation_type(self) -> TrajInterpolationType:
        """Interpolation that is actually executable on CPU/MPS.

        ``BSPLINE_KNOTS_CUDA`` remains the pinned public default.  Its raw
        CUDA kernel is unavailable, and the TrajOpt facade executes the
        endpoint-preserving linear portable fallback instead.  Keeping this
        conversion explicit gives callers a way to audit the boundary.
        """
        if self.interpolation_type == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
            return TrajInterpolationType.LINEAR_CUDA
        return self.interpolation_type

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
        for name, value in (("max_batch_size", max_batch_size), ("max_goalset", max_goalset),
                            ("num_seeds", num_seeds)):
            _positive_int(name, value)
        if not isinstance(multi_env, bool) or not isinstance(self_collision_check, bool):
            raise TypeError("multi_env and self_collision_check must be bool")
        num_envs = max_batch_size if multi_env else 1
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
        if robot_cfg.device_cfg != device_cfg:
            raise ValueError("robot DeviceCfg must match device_cfg")
        records = _copy_optimizer_configs(optimizer_configs)
        interpolation_type = _transition_interpolation_type(transition_model)
        optimizer_name = _optimizer_name(records)
        if override_optimizer_num_iters is not None and not isinstance(override_optimizer_num_iters, dict):
            raise TypeError("override_optimizer_num_iters must be a mapping or None")
        overrides = {} if override_optimizer_num_iters is None else deepcopy(override_optimizer_num_iters)
        for name, override in overrides.items():
            if not isinstance(name, str):
                raise TypeError("override_optimizer_num_iters keys must be strings")
            if override is not None:
                _positive_int(f"override_optimizer_num_iters[{name!r}]", override)
        max_iterations = overrides.get(optimizer_name)
        if max_iterations is None:
            max_iterations = 200
        scene_cfg = _scene_cfg(scene_model, collision_cache, device_cfg, num_envs)
        # These YAML-shaped records are valuable for inspection and later
        # upgrades, but are deliberately not posed as compiled CUDA/Warp
        # rollout objects.  TrajOptSolver owns its portable composed-Torch
        # objective directly.
        core = SolverCoreCfg(
            robot_cfg, device_cfg, records,
            optimizer_rollout_configs=[deepcopy(transition_model) for _ in records],
            metrics_rollout_config=deepcopy(metrics_rollout), scene_collision_cfg=scene_cfg,
            use_cuda_graph=use_cuda_graph, random_seed=random_seed, store_debug=store_debug,
        )
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
