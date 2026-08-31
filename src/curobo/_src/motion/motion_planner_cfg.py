"""Pinned MotionPlanner configuration compiled to portable solver configs.

The upstream factory is the public configuration boundary for most cuRobo
applications.  This module deliberately compiles paths and dictionaries at
that boundary instead of leaving task/scene values as opaque objects that
would fail much later in a solve call.  The resulting records are ordinary
CPU/MPS PyTorch configurations; CUDA graph capture remains an explicitly
unavailable execution optimization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import math
from os import PathLike
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config, resolve_device_cfg
from curobo._src.util.logging import log_and_raise
from curobo.content import get_robot_configs_path, get_scene_configs_path


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    """Validate capacities before they allocate planner-owned buffers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _resolve_scene(scene_model: Any) -> SceneCfg | list[SceneCfg] | SceneCollisionCfg:
    """Resolve the portable subset of V2 scene inputs.

    Scene YAML names are intentionally resolved relative to bundled scene
    content, just like robot names.  USD/Isaac values are rejected by
    :func:`resolve_config` with an explicit portable boundary.
    """
    if isinstance(scene_model, (SceneCfg, SceneCollisionCfg)):
        return scene_model
    if isinstance(scene_model, (str, PathLike)):
        scene_model = resolve_config(join_path(get_scene_configs_path(), scene_model))
    if isinstance(scene_model, Mapping):
        return SceneCfg.create(dict(scene_model))
    if isinstance(scene_model, list):
        resolved: list[SceneCfg] = []
        for index, value in enumerate(scene_model):
            if isinstance(value, (str, PathLike)):
                value = resolve_config(join_path(get_scene_configs_path(), value))
            if isinstance(value, SceneCfg):
                resolved.append(value)
            elif isinstance(value, Mapping):
                resolved.append(SceneCfg.create(dict(value)))
            else:
                raise TypeError(
                    "scene_model entries must be SceneCfg, mappings, or YAML paths; "
                    f"entry {index} is {type(value).__name__}"
                )
        if not resolved:
            raise ValueError("scene_model must not be an empty environment list")
        return resolved
    raise TypeError(
        "scene_model must be SceneCfg, a scene mapping, a YAML path, or a nonempty list"
    )


def _validate_collision_cache(cache: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if cache is None:
        return None
    if not isinstance(cache, Mapping):
        raise TypeError("collision_cache must be a mapping from obstacle kind to capacity")
    known = {"cuboid", "primitive", "obb", "mesh", "voxel"}
    unknown = set(cache).difference(known)
    if unknown:
        raise ValueError("unknown collision_cache key(s): " + ", ".join(sorted(unknown)))
    result: Dict[str, int] = {}
    for name, capacity in cache.items():
        result[str(name)] = _positive_int(capacity, f"collision_cache[{name!r}]", allow_zero=True)
    return result


class _MotionPlannerCfgPortableMixin:
    @property
    def robot_config(self) -> RobotCfg:
        return self._robot_config_portable

    @property
    def requested_use_cuda_graph(self) -> bool:
        return self._requested_use_cuda_graph_portable

    def clone(self, **updates: Any) -> "MotionPlannerCfg":
        return self._clone_portable(**updates)

    copy = clone

    def update(self, **updates: Any) -> "MotionPlannerCfg":
        return self._update_portable(**updates)


@dataclass
class MotionPlannerCfg(_MotionPlannerCfgPortableMixin):
    ik_solver_config: IKSolverCfg
    trajopt_solver_config: TrajOptSolverCfg
    graph_planner_config: Optional[PRMGraphPlannerCfg] = None
    scene_collision_cfg: Optional[SceneCollisionCfg] = None
    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self) -> None:
        """Validate the direct-construction form used by advanced callers.

        ``create`` is the normal public entry point, but cuRobo applications
        also compose pre-built IK, TrajOpt, and PRM records.  The planner shares
        their world and shape-bound assumptions, so accepting contradictory
        records here otherwise leads to a failure deep inside a solve call.
        The records may request CUDA graph capture, but their portable active
        setting remains ordinary eager/persistent CPU/MPS state.
        """
        self.device_cfg = resolve_device_cfg(self.device_cfg)
        if not isinstance(self.ik_solver_config, IKSolverCfg):
            raise TypeError("ik_solver_config must be IKSolverCfg")
        if not isinstance(self.trajopt_solver_config, TrajOptSolverCfg):
            raise TypeError("trajopt_solver_config must be TrajOptSolverCfg")
        if self.graph_planner_config is not None and not isinstance(
            self.graph_planner_config, PRMGraphPlannerCfg
        ):
            raise TypeError("graph_planner_config must be PRMGraphPlannerCfg or None")
        if self.scene_collision_cfg is not None and not isinstance(
            self.scene_collision_cfg, SceneCollisionCfg
        ):
            raise TypeError("scene_collision_cfg must be SceneCollisionCfg or None")

        children: list[tuple[str, Any]] = [
            ("ik_solver_config", self.ik_solver_config),
            ("trajopt_solver_config", self.trajopt_solver_config),
        ]
        if self.graph_planner_config is not None:
            children.append(("graph_planner_config", self.graph_planner_config))
        for name, config in children:
            if config.device_cfg != self.device_cfg:
                raise ValueError(f"{name}.device_cfg must match MotionPlannerCfg.device_cfg")

        ik, traj = self.ik_solver_config, self.trajopt_solver_config
        for name in ("max_batch_size", "multi_env", "max_goalset"):
            if getattr(ik, name) != getattr(traj, name):
                raise ValueError(
                    f"ik_solver_config and trajopt_solver_config must agree on {name}"
                )
        expected_envs = ik.max_batch_size if ik.multi_env else 1
        if self.scene_collision_cfg is not None:
            if self.scene_collision_cfg.device_cfg != self.device_cfg:
                raise ValueError("scene_collision_cfg.device_cfg must match MotionPlannerCfg.device_cfg")
            if self.scene_collision_cfg.num_envs != expected_envs:
                raise ValueError(
                    "scene_collision_cfg.num_envs must match the configured planner environments"
                )
            for name, config in (("ik_solver_config", ik), ("trajopt_solver_config", traj)):
                if config.scene_collision_cfg is not self.scene_collision_cfg:
                    raise ValueError(
                        f"{name}.scene_collision_cfg must be the planner scene_collision_cfg"
                    )
            if (
                self.graph_planner_config is not None
                and self.graph_planner_config.scene_collision_cfg is not self.scene_collision_cfg
            ):
                raise ValueError(
                    "graph_planner_config.scene_collision_cfg must be the planner scene_collision_cfg"
                )
        else:
            for name, config in (("ik_solver_config", ik), ("trajopt_solver_config", traj)):
                if config.scene_collision_cfg is not None:
                    raise ValueError(
                        f"{name}.scene_collision_cfg must be None when planner scene_collision_cfg is None"
                    )
            if (
                self.graph_planner_config is not None
                and self.graph_planner_config.scene_collision_cfg is not None
            ):
                raise ValueError(
                    "graph_planner_config.scene_collision_cfg must be None when planner scene_collision_cfg is None"
                )

    @property
    def _robot_config_portable(self) -> RobotCfg:
        """The robot configuration shared by the planner's portable solvers."""
        return self.ik_solver_config.robot_config

    @property
    def _requested_use_cuda_graph_portable(self) -> bool:
        """Whether the caller requested CUDA graph capture on construction.

        It is retained for source compatibility and diagnostics only.  The
        active ``use_cuda_graph`` fields of portable child records are always
        false because graph capture is a CUDA runtime ABI.
        """
        return bool(
            self.ik_solver_config.requested_use_cuda_graph
            or self.trajopt_solver_config.requested_use_cuda_graph
            or (
                self.graph_planner_config is not None
                and self.graph_planner_config.use_cuda_graph_for_rollout
            )
        )

    def _clone_portable(self, **updates: Any) -> "MotionPlannerCfg":
        """Return a validated independent configuration tree.

        Deep copying the complete aggregation preserves aliases between a
        shared scene and each child core, while isolating mutable optimizer,
        cache, and tensor-bound records from the source configuration.  Device
        migration is intentionally not implicit: robot/world tensors must be
        rebuilt through :meth:`create` for their target ``DeviceCfg``.
        """
        known = {item.name for item in fields(self)}
        unknown = sorted(set(updates).difference(known))
        if unknown:
            raise TypeError(f"unknown MotionPlannerCfg fields: {unknown}")
        if "device_cfg" in updates and resolve_device_cfg(updates["device_cfg"]) != self.device_cfg:
            raise ValueError(
                "MotionPlannerCfg cannot migrate prebuilt robot/world tensors; "
                "create a new configuration for the target DeviceCfg"
            )
        # ``deepcopy`` preserves aliases in the original aggregate. When a
        # caller replaces either a child config or the world, this aggregate
        # remains the authority for world ownership: rewire all three children
        # to its copied/replaced scene before complete validation.
        candidate = deepcopy(self)
        for name, value in updates.items():
            setattr(candidate, name, value)
        scene_or_child_update = {
            "scene_collision_cfg",
            "ik_solver_config",
            "trajopt_solver_config",
            "graph_planner_config",
        }
        if scene_or_child_update.intersection(updates):
            canonical_scene = candidate.scene_collision_cfg
            for name in ("ik_solver_config", "trajopt_solver_config"):
                config = getattr(candidate, name)
                if config.scene_collision_cfg is not canonical_scene:
                    core = config.core_cfg.clone(scene_collision_cfg=canonical_scene)
                    setattr(
                        candidate,
                        name,
                        config.clone(core_cfg=core, robot_config=core.robot_config),
                    )
            if (
                candidate.graph_planner_config is not None
                and candidate.graph_planner_config.scene_collision_cfg is not canonical_scene
            ):
                candidate.graph_planner_config = candidate.graph_planner_config.clone(
                    scene_collision_cfg=canonical_scene
                )
        candidate.__post_init__()
        return candidate

    def _update_portable(self, **updates: Any) -> "MotionPlannerCfg":
        """Atomically replace fields after validating the complete configuration."""
        candidate = self.clone(**updates)
        for item in fields(self):
            setattr(self, item.name, getattr(candidate, item.name))
        return self

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"][1:],
        ik_transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        trajopt_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"],
        trajopt_transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        graph_planner_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        graph_planner_transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: int = 1000,
    ) -> MotionPlannerCfg:
        """Compile a V2-shaped planner request into portable typed configs.

        ``use_cuda_graph`` is accepted so applications can retain their V2
        call sites.  It creates no CUDA graph on CPU/MPS; portable solvers use
        persistent shape-keyed state instead.  Explicit graph manipulation is
        still rejected by the solver APIs.
        """
        device_cfg = resolve_device_cfg(device_cfg)
        max_batch_size = _positive_int(max_batch_size, "max_batch_size")
        max_goalset = _positive_int(max_goalset, "max_goalset")
        interpolation_buffer_size = _positive_int(
            interpolation_buffer_size, "interpolation_buffer_size"
        )
        position_tolerance = _positive_scalar(position_tolerance, "position_tolerance")
        orientation_tolerance = _positive_scalar(orientation_tolerance, "orientation_tolerance")
        optimizer_collision_activation_distance = _positive_scalar(
            optimizer_collision_activation_distance, "optimizer_collision_activation_distance"
        )
        interpolation_dt = _positive_scalar(interpolation_dt, "interpolation_dt")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise TypeError("random_seed must be an integer")
        if not isinstance(self_collision_check, bool):
            raise TypeError("self_collision_check must be a bool")
        if not isinstance(multi_env, bool):
            raise TypeError("multi_env must be a bool")
        if not isinstance(store_debug, bool) or not isinstance(use_cuda_graph, bool):
            raise TypeError("store_debug and use_cuda_graph must be bool values")
        if num_ik_seeds is None:
            num_ik_seeds = 16 if max_batch_size > 1 else 32
        else:
            num_ik_seeds = _positive_int(num_ik_seeds, "num_ik_seeds")
        if num_trajopt_seeds is None:
            num_trajopt_seeds = 2 if max_batch_size > 1 else 4
        else:
            num_trajopt_seeds = _positive_int(num_trajopt_seeds, "num_trajopt_seeds")
        if not isinstance(ik_optimizer_configs, list) or not ik_optimizer_configs:
            raise ValueError("ik_optimizer_configs must be a nonempty list")
        if not isinstance(trajopt_optimizer_configs, list) or not trajopt_optimizer_configs:
            raise ValueError("trajopt_optimizer_configs must be a nonempty list")

        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        else:
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)

        # A typed SceneCollisionCfg lets direct IKSolver/SolverCore users and
        # high-level MotionPlanner users observe the same world/cache request.
        # It also makes multi-environment capacity visible before planning.
        collision_cache = _validate_collision_cache(collision_cache)
        scene_collision_cfg: Optional[SceneCollisionCfg] = None
        if scene_model is not None:
            scene = _resolve_scene(scene_model)
            if isinstance(scene, SceneCollisionCfg):
                if scene.device_cfg != device_cfg:
                    raise ValueError("scene_model SceneCollisionCfg device_cfg must match device_cfg")
                if collision_cache is not None and scene.cache != collision_cache:
                    raise ValueError(
                        "collision_cache cannot override a prebuilt SceneCollisionCfg cache"
                    )
                expected_envs = max_batch_size if multi_env else 1
                if scene.num_envs != expected_envs:
                    raise ValueError(
                        "scene_model SceneCollisionCfg.num_envs must match configured planner environments"
                    )
                scene_collision_cfg = scene
            elif isinstance(scene, list):
                if not multi_env:
                    raise ValueError("a list scene_model requires multi_env=True")
                if len(scene) != max_batch_size:
                    raise ValueError(
                        "multi_env scene_model length must equal max_batch_size "
                        f"({max_batch_size}), got {len(scene)}"
                    )
            if scene_collision_cfg is None:
                scene_collision_cfg = SceneCollisionCfg(
                    device_cfg=device_cfg,
                    scene_model=scene,
                    num_envs=max_batch_size if multi_env else 1,
                    cache=collision_cache,
                )
        elif collision_cache is not None:
            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=None,
                num_envs=max_batch_size if multi_env else 1,
                cache=collision_cache,
            )

        # CUDA graph capture has no equivalent on Metal.  ``store_debug`` has
        # the upstream effect of disabling capture; both cases map to ordinary
        # persistent portable buffers rather than a fake CUDA graph object.
        portable_graph_state = False if store_debug else use_cuda_graph
        ik_cfg = IKSolverCfg.create(
            robot_cfg, optimizer_configs=ik_optimizer_configs,
            transition_model=ik_transition_model,
            metrics_rollout=metrics_rollout,
            scene_model=scene_collision_cfg, collision_cache=collision_cache,
            self_collision_check=self_collision_check,
            device_cfg=device_cfg, num_seeds=num_ik_seeds or 32,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=portable_graph_state, random_seed=random_seed,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            store_debug=store_debug, max_batch_size=max_batch_size,
            multi_env=multi_env, max_goalset=max_goalset,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )
        traj_cfg = TrajOptSolverCfg.create(
            robot_cfg, optimizer_configs=trajopt_optimizer_configs,
            transition_model=trajopt_transition_model,
            metrics_rollout=metrics_rollout,
            scene_model=scene_collision_cfg, collision_cache=collision_cache,
            self_collision_check=self_collision_check,
            device_cfg=device_cfg, num_seeds=num_trajopt_seeds or 4,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=portable_graph_state, random_seed=random_seed,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            store_debug=store_debug,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=interpolation_buffer_size,
            max_batch_size=max_batch_size, multi_env=multi_env,
            max_goalset=max_goalset,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )
        graph_cfg = None
        if graph_planner_config is not None:
            graph_cfg = PRMGraphPlannerCfg.create(
                robot_cfg,
                graph_planner_config=graph_planner_config,
                rollout=graph_planner_rollout,
                transition_model=graph_planner_transition_model,
                scene_model=scene_collision_cfg,
                collision_cache=collision_cache,
                self_collision_check=self_collision_check,
                device_cfg=device_cfg,
                use_cuda_graph_for_rollout=portable_graph_state,
                transition_model_config_instance_type=transition_model_config_instance_type,
                cost_manager_config_instance_type=cost_manager_config_instance_type,
            )
        return MotionPlannerCfg(
            ik_solver_config=ik_cfg,
            trajopt_solver_config=traj_cfg,
            graph_planner_config=graph_cfg,
            scene_collision_cfg=scene_collision_cfg,
            device_cfg=device_cfg,
        )


__all__ = [
    "MotionPlannerCfg",
    "DeviceCfg",
    "IKSolverCfg",
    "PRMGraphPlannerCfg",
    "RobotCfg",
    "RobotCostManagerCfg",
    "RobotStateTransitionCfg",
    "SceneCfg",
    "SceneCollisionCfg",
    "TrajOptSolverCfg",
    "get_robot_configs_path",
    "get_scene_configs_path",
    "join_path",
    "log_and_raise",
    "resolve_config",
]
