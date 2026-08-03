"""Validated portable configuration for the pinned V2 PRM planner.

The upstream record is usually created indirectly from three task YAML files.
Those files configure CUDA rollout objects in cuRobo; on CPU/MPS we compile
the portable, observable portion instead: a deterministic sampler, a bounded
roadmap, a typed rollout record, and (when supplied) a portable scene
collision configuration.  CUDA graph capture remains an accepted *setting*,
not an object that this module pretends to create.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, Union

import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config, resolve_device_cfg
from curobo.content import get_robot_configs_path, get_scene_configs_path, get_task_configs_path


# The compact Metal distribution intentionally does not ship NVIDIA's task
# YAML corpus.  These are the upstream factory defaults; treating their absent
# paths as the class defaults preserves normal ``PRMGraphPlannerCfg.create``
# usage while any other missing path remains a real configuration error.
_BUNDLED_DEFAULTS = frozenset(
    {
        "graph_planner/exact_graph_planner.yml",
        "metrics_base.yml",
        "graph_planner/transition_graph_planner.yml",
    }
)
_ELLIPSOID_METHODS = frozenset({"svd", "householder", "approximate"})


def _finite_scalar(value: Any, name: str, *, minimum: float = 0.0, strict: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        qualifier = f"greater than {minimum}" if strict else f"at least {minimum}"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _resolve_task(value: Any, *, name: str) -> Any:
    """Resolve a task mapping/path without silently dropping user input."""
    if not isinstance(value, (str, PathLike)):
        return value
    path = Path(value)
    if path.is_absolute() or path.exists():
        return resolve_config(path)
    candidate = Path(join_path(get_task_configs_path(), path))
    if candidate.exists():
        return resolve_config(candidate)
    if str(value) in _BUNDLED_DEFAULTS:
        return {}
    raise FileNotFoundError(
        f"{name} configuration {value!r} was not found; the portable package only "
        "falls back for the pinned factory defaults"
    )


def _resolve_robot(
    value: Union[str, PathLike[str], Dict[str, Any], RobotCfg], device_cfg: DeviceCfg
) -> RobotCfg:
    """Resolve the pinned short robot-name convention before loading it.

    The V2 graph planner operates on the reduced c-space (for example the
    packaged Franka's locked finger joints are absent).  Reusing
    ``KinematicsCfg`` here ensures direct PRM construction has the same joint
    count as MotionPlannerCfg instead of exposing loader-internal fixed axes.
    """
    if isinstance(value, RobotCfg):
        return value
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    if isinstance(value, (str, PathLike)):
        candidate = Path(value)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(join_path(get_robot_configs_path(), candidate))
        kin = KinematicsCfg.from_robot_yaml_file(str(candidate), device_cfg=device_cfg)
    else:
        kin = KinematicsCfg.from_robot_yaml_file(value, device_cfg=device_cfg)
    model = kin.kinematics_config.robot_cfg
    return RobotCfg(model, dynamics=model.dynamics, device_cfg=device_cfg)


def _mapping_section(value: Any, section: str, *, name: str) -> dict[str, Any]:
    """Extract a conventional YAML section and return an independent mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must resolve to a mapping, got {type(value).__name__}")
    raw = value.get(section, value)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name}.{section} must be a mapping")
    return dict(raw)


def _resolve_scene(
    value: Any, device_cfg: DeviceCfg, cache: Optional[Dict[str, int]]
) -> Optional[SceneCollisionCfg]:
    if value is None:
        return None
    if isinstance(value, SceneCollisionCfg):
        if value.device_cfg != device_cfg:
            # Scene records own their buffers; silently replacing the device is
            # unsafe.  Callers can construct a matching SceneCollisionCfg.
            raise ValueError("scene_collision_cfg device_cfg must match PRM device_cfg")
        return value
    if isinstance(value, (str, PathLike)):
        candidate = Path(value)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(join_path(get_scene_configs_path(), candidate))
        value = resolve_config(candidate)
    if isinstance(value, Mapping):
        return SceneCollisionCfg(
            device_cfg=device_cfg, scene_model=SceneCfg.create(dict(value)), cache=cache
        )
    if isinstance(value, list):
        scenes = [item if isinstance(item, SceneCfg) else SceneCfg.create(dict(item)) for item in value]
        return SceneCollisionCfg(device_cfg=device_cfg, scene_model=scenes, cache=cache)
    if isinstance(value, SceneCfg):
        return SceneCollisionCfg(device_cfg=device_cfg, scene_model=value, cache=cache)
    raise TypeError("scene_model must be SceneCfg, mapping, scene YAML path, list, or SceneCollisionCfg")


def _validate_collision_cache(cache: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if cache is None:
        return None
    if not isinstance(cache, Mapping):
        raise TypeError("collision_cache must be a mapping from obstacle kind to capacity")
    supported = {"cuboid", "primitive", "obb", "mesh", "voxel"}
    unknown = set(cache).difference(supported)
    if unknown:
        raise ValueError("unknown collision_cache key(s): " + ", ".join(sorted(map(str, unknown))))
    result: Dict[str, int] = {}
    for kind, capacity in cache.items():
        result[str(kind)] = _integer(capacity, f"collision_cache[{kind!r}]", minimum=0)
    return result


@dataclass
class PRMGraphPlannerCfg:
    # These defaults match the prior portable factory and make the record
    # convenient to use directly in Python, unlike upstream's YAML-only form.
    max_nodes: int = 2048
    feasibility_buffer_size: int = 4096
    steer_buffer_size: int = 4096
    exploration_radius: float = 2.0
    new_nodes_per_iteration: int = 128
    max_path_finding_iterations: int = 20
    min_finetune_iterations: int = 0
    use_default_position_heuristic: bool = True
    cspace_similarity_threshold: float = 1e-4
    sample_rejection_ratio: int = 10
    neighbors_per_node: int = 12
    rollout_config: Any = None
    sampler_seed: int = 0
    sampler_buffer_size: int = 4096
    use_cuda_graph_for_rollout: bool = True
    connect_terminal_nodes_with_nearest: bool = True
    exploration_radius_growth_factor: float = 1.5
    neighbors_per_node_growth_factor: float = 1.5
    new_nodes_per_iteration_growth_factor: float = 1.5
    ellipsoid_projection_method: str = "householder"
    scene_collision_cfg: Optional[Any] = None
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    graph_path_finder_seed: int = 42
    # Portable execution data; additive fields do not alter V2 constructor
    # compatibility, but let the production graph operator run without CUDA.
    robot_config: Optional[RobotCfg] = None
    action_lower_bounds: Optional[torch.Tensor] = None
    action_upper_bounds: Optional[torch.Tensor] = None
    check_feasibility_fn: Any = None
    connection_radius: Optional[float] = None
    edge_step: float = 0.05
    self_collision_check: bool = True

    def __post_init__(self) -> None:
        self.device_cfg = resolve_device_cfg(self.device_cfg)
        self.max_nodes = _integer(self.max_nodes, "max_nodes", minimum=2)
        self.feasibility_buffer_size = _integer(
            self.feasibility_buffer_size, "feasibility_buffer_size", minimum=1
        )
        self.steer_buffer_size = _integer(self.steer_buffer_size, "steer_buffer_size", minimum=1)
        self.new_nodes_per_iteration = _integer(
            self.new_nodes_per_iteration, "new_nodes_per_iteration", minimum=0
        )
        self.max_path_finding_iterations = _integer(
            self.max_path_finding_iterations, "max_path_finding_iterations", minimum=0
        )
        self.min_finetune_iterations = _integer(
            self.min_finetune_iterations, "min_finetune_iterations", minimum=0
        )
        self.sample_rejection_ratio = _integer(self.sample_rejection_ratio, "sample_rejection_ratio", minimum=1)
        self.neighbors_per_node = _integer(self.neighbors_per_node, "neighbors_per_node", minimum=1)
        self.sampler_buffer_size = _integer(
            self.sampler_buffer_size, "sampler_buffer_size", minimum=1
        )
        for name in ("sampler_seed", "graph_path_finder_seed"):
            setattr(self, name, _integer(getattr(self, name), name))
        self.exploration_radius = _finite_scalar(self.exploration_radius, "exploration_radius")
        self.cspace_similarity_threshold = _finite_scalar(
            self.cspace_similarity_threshold, "cspace_similarity_threshold", strict=False
        )
        self.edge_step = _finite_scalar(self.edge_step, "edge_step")
        for name in (
            "exploration_radius_growth_factor", "neighbors_per_node_growth_factor",
            "new_nodes_per_iteration_growth_factor",
        ):
            setattr(self, name, _finite_scalar(getattr(self, name), name, minimum=1.0, strict=False))
        if self.connection_radius is not None:
            self.connection_radius = _finite_scalar(self.connection_radius, "connection_radius")
        for name in (
            "use_default_position_heuristic",
            "use_cuda_graph_for_rollout",
            "connect_terminal_nodes_with_nearest",
            "self_collision_check",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.ellipsoid_projection_method not in _ELLIPSOID_METHODS:
            raise ValueError(
                "ellipsoid_projection_method must be one of "
                + ", ".join(sorted(_ELLIPSOID_METHODS))
            )
        if self.check_feasibility_fn is not None and not callable(self.check_feasibility_fn):
            raise TypeError("check_feasibility_fn must be callable or None")
        self._canonicalize_bounds()
        if self.scene_collision_cfg is not None and not isinstance(
            self.scene_collision_cfg, SceneCollisionCfg
        ):
            raise TypeError("scene_collision_cfg must be a SceneCollisionCfg or None")
        if self.scene_collision_cfg is not None and self.scene_collision_cfg.device_cfg != self.device_cfg:
            raise ValueError("scene_collision_cfg device_cfg must match PRM device_cfg")
        if self.rollout_config is not None:
            rollout_device = getattr(self.rollout_config, "device_cfg", self.device_cfg)
            if rollout_device != self.device_cfg:
                raise ValueError("rollout_config device_cfg must match PRM device_cfg")
        if self.robot_config is not None:
            robot_device = getattr(self.robot_config, "device_cfg", self.device_cfg)
            if robot_device != self.device_cfg:
                raise ValueError("robot_config device_cfg must match PRM device_cfg")

    def _canonicalize_bounds(self) -> None:
        lower, upper = self.action_lower_bounds, self.action_upper_bounds
        if (lower is None) != (upper is None):
            raise ValueError(
                "action_lower_bounds and action_upper_bounds must be supplied together"
            )
        if lower is None:
            return
        lower = torch.as_tensor(lower, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        upper = torch.as_tensor(upper, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape or lower.numel() == 0:
            raise ValueError("action bounds must be matching nonempty rank-1 tensors")
        if not bool(torch.isfinite(lower).all().item()) or not bool(torch.isfinite(upper).all().item()):
            raise ValueError("action bounds must contain finite values")
        if not bool((lower < upper).all().item()):
            raise ValueError("every action lower bound must be smaller than its upper bound")
        self.action_lower_bounds, self.action_upper_bounds = lower, upper

    @property
    def action_dim(self) -> Optional[int]:
        """The compiled c-space dimension, or ``None`` before robot/bounds input."""
        return None if self.action_lower_bounds is None else int(self.action_lower_bounds.numel())

    def clone(self, **updates: Any) -> "PRMGraphPlannerCfg":
        """Return a validated copy suitable for independent persistent planners."""
        values = {
            name: copy.copy(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        for name in ("action_lower_bounds", "action_upper_bounds"):
            if values[name] is not None:
                values[name] = values[name].clone()
        values["device_cfg"] = self.device_cfg.clone()
        values.update(updates)
        return type(self)(**values)

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        use_cuda_graph_for_rollout: bool = True,
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        graph_path_finder_seed: int = 42,
    ) -> "PRMGraphPlannerCfg":
        """Compile V2-style flexible config inputs into a portable PRM record.

        The object accepts the same paths/mappings used by V2.  Supplied YAML
        is validated rather than ignored.  The three canonical task defaults
        use portable class defaults when the optional NVIDIA task corpus is not
        installed.  CUDA graph settings are retained for API compatibility but
        are represented by ordinary persistent CPU/MPS execution state.
        """
        device_cfg = resolve_device_cfg(device_cfg)
        if not isinstance(self_collision_check, bool):
            raise TypeError("self_collision_check must be bool")
        if not isinstance(use_cuda_graph_for_rollout, bool):
            raise TypeError("use_cuda_graph_for_rollout must be bool")
        collision_cache = _validate_collision_cache(collision_cache)
        robot_cfg = _resolve_robot(robot, device_cfg)
        graph_data = _mapping_section(
            _resolve_task(graph_planner_config, name="graph_planner"),
            "graph_planner",
            name="graph_planner",
        )
        rollout_data = _mapping_section(
            _resolve_task(rollout, name="rollout"), "rollout", name="rollout"
        )
        transition_data = _mapping_section(
            _resolve_task(transition_model, name="transition_model"),
            "transition_model_cfg",
            name="transition_model",
        )
        # Match V2's factory ordering: the separately supplied transition
        # configuration deliberately replaces any rollout-file placeholder.
        rollout_data["transition_model_cfg"] = transition_data or None
        # The simple portable rollout type deliberately ignores CUDA-only
        # sub-components but preserves its typed lifecycle and selected seed.
        rollout_cfg = RobotRolloutCfg.create_with_component_types(
            rollout_data,
            robot_cfg,
            device_cfg=device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )
        forbidden = {
            "device_cfg",
            "scene_collision_cfg",
            "self_collision_check",
            "rollout_config",
            "robot_config",
            "action_lower_bounds",
            "action_upper_bounds",
            "graph_path_finder_seed",
        }
        overlap = forbidden.intersection(graph_data)
        if overlap:
            raise ValueError(
                "graph_planner configuration cannot override factory-owned field(s): "
                + ", ".join(sorted(overlap))
            )
        joints = [
            joint
            for joint in robot_cfg.kinematics.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        if not joints:
            raise ValueError("robot must expose at least one active non-mimic joint for PRM planning")
        lower = device_cfg.to_device([joint.limits.lower for joint in joints])
        upper = device_cfg.to_device([joint.limits.upper for joint in joints])
        scene_cfg = _resolve_scene(scene_model, device_cfg, collision_cache)
        return PRMGraphPlannerCfg(
            **graph_data,
            rollout_config=rollout_cfg,
            robot_config=robot_cfg,
            action_lower_bounds=lower,
            action_upper_bounds=upper,
            scene_collision_cfg=scene_cfg,
            self_collision_check=self_collision_check,
            device_cfg=device_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph_for_rollout,
            graph_path_finder_seed=graph_path_finder_seed,
        )


__all__ = ["PRMGraphPlannerCfg"]
