from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import resolve_device_cfg


def _is_portable_scene_model(value: Any) -> bool:
    """Return whether ``value`` is an already-compiled portable world.

    Importing the collision records lazily keeps this lightweight config module
    usable in applications that only construct retargeting YAML records.  The
    retargeter itself routes these records through its real IK/MPC collision
    backends; CUDA/Warp scene objects are deliberately not accepted by name.
    """
    from curobo._src.geom.collision.collision_scene import (
        SceneCollision,
        SceneCollisionCfg,
    )
    from curobo._src.geom.types import SceneCfg

    return isinstance(value, (SceneCfg, SceneCollisionCfg, SceneCollision))


def _criterion_on_device(
    criterion: ToolPoseCriteria, device_cfg: DeviceCfg
) -> ToolPoseCriteria:
    """Copy one criterion into a retargeter's declared tensor domain.

    V2 configuration examples normally construct criteria with the global
    device configuration.  That is easy to miss when a caller supplies an
    explicit MPS ``DeviceCfg`` though: the criterion factories default to
    CPU.  Letting those CPU tensors reach the IK cost produces a late and
    opaque mixed-device error.  Retargeter configuration is a natural
    compilation boundary, so make its own immutable copy on the requested
    device instead.  This also means one reusable criteria dictionary can
    safely configure a CPU and an MPS retargeter independently.
    """
    target = device_cfg.device
    return ToolPoseCriteria(
        criterion.terminal_pose_axes_weight_factor.to(
            device=target, dtype=device_cfg.dtype
        ),
        criterion.non_terminal_pose_axes_weight_factor.to(
            device=target, dtype=device_cfg.dtype
        ),
        criterion.terminal_pose_convergence_tolerance.to(
            device=target, dtype=device_cfg.dtype
        ),
        criterion.non_terminal_pose_convergence_tolerance.to(
            device=target, dtype=device_cfg.dtype
        ),
        criterion.project_distance_to_goal.to(device=target),
        device_cfg,
    )


def _optimizer_configs(
    value: Any, name: str
) -> List[Union[str, Dict[str, Any]]]:
    """Validate and detach an ordered V2 optimizer configuration sequence."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of YAML paths or mappings")
    compiled: List[Union[str, Dict[str, Any]]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"{name}[{index}] cannot be an empty YAML path")
            compiled.append(item)
        elif isinstance(item, Mapping):
            # A fresh top-level mapping prevents a solver configuration
            # builder from altering a caller-owned input while preserving the
            # nested task/cost payload used by the V2 YAML vocabulary.
            compiled.append(dict(item))
        else:
            raise TypeError(
                f"{name}[{index}] must be a YAML path or mapping, got "
                f"{type(item).__name__}"
            )
    return compiled


@dataclass
class MotionRetargeterCfg:
    robot: Union[str, Dict[str, Any]]
    tool_pose_criteria: Dict[str, ToolPoseCriteria]
    num_envs: int = 1
    use_mpc: bool = False
    self_collision_check: bool = True
    scene_model: Optional[Any] = None
    optimization_dt: float = 0.05
    num_seeds_global: int = 64
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    load_collision_spheres: bool = True
    ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["ik/lbfgs_retarget_ik.yml"]
    )
    mpc_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["mpc/lbfgs_retarget_mpc.yml"]
    )
    num_seeds_local: int = 1
    num_control_points: Optional[int] = None
    steps_per_target: int = 8
    velocity_regularization_weight: Optional[float] = None
    acceleration_regularization_weight: Optional[float] = None
    collision_activation_distance: float = 0.01
    global_ik_num_iters: Optional[int] = None
    local_ik_num_iters: Optional[int] = None
    mpc_warm_start_num_iters: int = 100
    mpc_cold_start_num_iters: int = 300

    @property
    def tool_frames(self) -> List[str]:
        """Tracked-link names in the deterministic criteria insertion order."""
        return list(self.tool_pose_criteria)

    @staticmethod
    def create(
        robot, tool_pose_criteria, num_envs=1, use_mpc=False,
        self_collision_check=True, scene_model=None, optimization_dt=0.05,
        num_seeds_global=64, load_collision_spheres=True,
        ik_optimizer_configs=None, mpc_optimizer_configs=None,
        num_seeds_local=1, num_control_points=12, steps_per_target=4,
        position_tolerance=0.005, orientation_tolerance=0.05,
        device_cfg=DeviceCfg(), velocity_regularization_weight=None,
        acceleration_regularization_weight=None,
        collision_activation_distance=0.01, global_ik_num_iters=None,
        local_ik_num_iters=None, mpc_warm_start_num_iters=100,
        mpc_cold_start_num_iters=300,
    ):
        # Keep the pinned V2 ``None``-selects-default distinction.  In
        # particular, an explicitly empty optimizer list is a configuration
        # error rather than an accidental request for the default YAML.
        return MotionRetargeterCfg(
            robot=robot,
            tool_pose_criteria=tool_pose_criteria,
            num_envs=num_envs,
            use_mpc=use_mpc,
            self_collision_check=self_collision_check,
            scene_model=scene_model,
            optimization_dt=optimization_dt,
            num_seeds_global=num_seeds_global,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres,
            ik_optimizer_configs=(
                ["ik/lbfgs_retarget_ik.yml"]
                if ik_optimizer_configs is None else ik_optimizer_configs
            ),
            mpc_optimizer_configs=(
                ["mpc/lbfgs_retarget_mpc.yml"]
                if mpc_optimizer_configs is None else mpc_optimizer_configs
            ),
            num_seeds_local=num_seeds_local,
            num_control_points=num_control_points,
            steps_per_target=steps_per_target,
            velocity_regularization_weight=velocity_regularization_weight,
            acceleration_regularization_weight=acceleration_regularization_weight,
            collision_activation_distance=collision_activation_distance,
            global_ik_num_iters=global_ik_num_iters,
            local_ik_num_iters=local_ik_num_iters,
            mpc_warm_start_num_iters=mpc_warm_start_num_iters,
            mpc_cold_start_num_iters=mpc_cold_start_num_iters,
        )

    def __post_init__(self):
        self.device_cfg = resolve_device_cfg(self.device_cfg)
        if not isinstance(self.robot, (str, Mapping)):
            raise TypeError("robot must be a YAML path or configuration mapping")
        if isinstance(self.robot, str) and not self.robot.strip():
            raise ValueError("robot cannot be an empty YAML path")
        if isinstance(self.robot, Mapping):
            if not self.robot:
                raise ValueError("robot configuration mapping cannot be empty")
            self.robot = dict(self.robot)
        if self.scene_model is not None and not isinstance(self.scene_model, (str, Mapping)) \
                and not _is_portable_scene_model(self.scene_model):
            raise TypeError(
                "scene_model must be a YAML path, configuration mapping, SceneCfg, "
                "SceneCollisionCfg, SceneCollision, or None"
            )
        if isinstance(self.scene_model, str) and not self.scene_model.strip():
            raise ValueError("scene_model cannot be an empty YAML path")
        if isinstance(self.scene_model, Mapping):
            if not self.scene_model:
                raise ValueError("scene_model configuration mapping cannot be empty")
            self.scene_model = dict(self.scene_model)
        for name in ("use_mpc", "self_collision_check", "load_collision_spheres"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.tool_pose_criteria, Mapping):
            raise TypeError("tool_pose_criteria must be a mapping from link name to ToolPoseCriteria")
        if not self.tool_pose_criteria:
            raise ValueError("tool_pose_criteria cannot be empty")
        compiled_criteria: Dict[str, ToolPoseCriteria] = {}
        for name, criterion in self.tool_pose_criteria.items():
            if not isinstance(name, str) or not name:
                raise ValueError("tool_pose_criteria link names must be nonempty strings")
            if not isinstance(criterion, ToolPoseCriteria):
                raise TypeError("tool_pose_criteria values must be ToolPoseCriteria")
            compiled_criteria[name] = _criterion_on_device(criterion, self.device_cfg)
        # ``dict`` preserves the caller's insertion order, which remains the
        # public V2 tool-frame order and the cost tensor's link dimension.
        self.tool_pose_criteria = compiled_criteria
        self.ik_optimizer_configs = _optimizer_configs(
            self.ik_optimizer_configs, "ik_optimizer_configs"
        )
        self.mpc_optimizer_configs = _optimizer_configs(
            self.mpc_optimizer_configs, "mpc_optimizer_configs"
        )
        integer_fields = {
            "num_envs": self.num_envs,
            "num_seeds_global": self.num_seeds_global,
            "num_seeds_local": self.num_seeds_local,
            "steps_per_target": self.steps_per_target,
            "mpc_warm_start_num_iters": self.mpc_warm_start_num_iters,
            "mpc_cold_start_num_iters": self.mpc_cold_start_num_iters,
        }
        if self.num_control_points is not None:
            integer_fields["num_control_points"] = self.num_control_points
        invalid = [name for name, value in integer_fields.items()
                   if isinstance(value, bool) or not isinstance(value, int) or value <= 0]
        if invalid:
            raise ValueError(f"retargeter integer fields must be positive: {invalid}")
        scalar_fields = {
            "optimization_dt": self.optimization_dt,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "collision_activation_distance": self.collision_activation_distance,
        }
        for name, value in scalar_fields.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("velocity_regularization_weight", "acceleration_regularization_weight"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative when provided")
        for name in ("global_ik_num_iters", "local_ik_num_iters"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when provided")
        if self.mpc_cold_start_num_iters < self.mpc_warm_start_num_iters:
            raise ValueError(
                "mpc_cold_start_num_iters must be greater than or equal to "
                "mpc_warm_start_num_iters"
            )
        # This follows V2's observable configuration contract.  It is useful
        # on Metal too: callers that explicitly disabled all collision checks
        # do not pay to load a sphere model.
        if not self.self_collision_check and self.scene_model is None:
            self.load_collision_spheres = False
        if self.self_collision_check and not self.load_collision_spheres:
            raise ValueError("load_collision_spheres must be True when self_collision_check=True")
        if self.scene_model is not None and not self.load_collision_spheres:
            raise ValueError("load_collision_spheres must be True when scene_model is set")

    @property
    def batch_shape(self) -> tuple[int, int]:
        """Declared ``[environment, tracked-tool]`` planning capacity.

        This is a pure configuration value: it does not allocate CUDA-graph
        buffers.  It is useful to validate goal tensors before constructing a
        stateful portable retargeter on CPU or MPS.
        """
        return self.num_envs, len(self.tool_pose_criteria)

    def with_device(self, device_cfg: DeviceCfg | str | Mapping[str, Any]) -> "MotionRetargeterCfg":
        """Recompile an equivalent configuration for CPU or MPS.

        The original criteria remain untouched.  CUDA is rejected by
        :func:`resolve_device_cfg`, so this method cannot accidentally create
        a configuration that silently falls back to CPU on a Metal host.
        """
        return self._copy_with(device_cfg=resolve_device_cfg(device_cfg))

    def with_tool_pose_criteria(
        self, tool_pose_criteria: Mapping[str, ToolPoseCriteria]
    ) -> "MotionRetargeterCfg":
        """Return a recompiled configuration with new per-tool criteria.

        The replacement is compiled into this configuration's device and
        dtype, retaining the source mapping as caller-owned data.  A running
        :class:`MotionRetargeter` accepts only the same ordered tool topology;
        changing tracked links requires a fresh solver because its cost and
        collision layout are shape-dependent.
        """
        return self._copy_with(tool_pose_criteria=tool_pose_criteria)

    def with_scene_model(self, scene_model: Optional[Any]) -> "MotionRetargeterCfg":
        """Return an equivalent configuration referring to a portable world.

        This is useful after :meth:`MotionRetargeter.update_world`; the
        collision object itself is intentionally retained, rather than
        serialized into a lossy YAML-shaped mapping.
        """
        return self._copy_with(scene_model=scene_model)

    def _copy_with(self, **updates: Any) -> "MotionRetargeterCfg":
        """Build a fully validated independent config without aliasing lists."""
        values = dict(
            robot=self.robot,
            tool_pose_criteria=self.tool_pose_criteria,
            num_envs=self.num_envs,
            use_mpc=self.use_mpc,
            self_collision_check=self.self_collision_check,
            scene_model=self.scene_model,
            optimization_dt=self.optimization_dt,
            num_seeds_global=self.num_seeds_global,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            device_cfg=self.device_cfg,
            load_collision_spheres=self.load_collision_spheres,
            ik_optimizer_configs=self.ik_optimizer_configs,
            mpc_optimizer_configs=self.mpc_optimizer_configs,
            num_seeds_local=self.num_seeds_local,
            num_control_points=self.num_control_points,
            steps_per_target=self.steps_per_target,
            velocity_regularization_weight=self.velocity_regularization_weight,
            acceleration_regularization_weight=self.acceleration_regularization_weight,
            collision_activation_distance=self.collision_activation_distance,
            global_ik_num_iters=self.global_ik_num_iters,
            local_ik_num_iters=self.local_ik_num_iters,
            mpc_warm_start_num_iters=self.mpc_warm_start_num_iters,
            mpc_cold_start_num_iters=self.mpc_cold_start_num_iters,
        )
        unknown = sorted(set(updates).difference(values))
        if unknown:
            raise TypeError(f"unknown MotionRetargeterCfg fields: {unknown}")
        values.update(updates)
        return type(self)(**values)


__all__ = ["MotionRetargeterCfg"]
