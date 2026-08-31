"""Pinned robot-scene collision configuration record."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from typing import Any, Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.cost.portable import (
    CSpaceCostCfg,
    CSpaceCostType,
    PositionCSpaceCost,
    SceneCollisionCost,
    SceneCollisionCostCfg,
    SelfCollisionCost,
    SelfCollisionCostCfg,
)
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.robot.types import SelfCollisionKinematicsCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util_file import get_robot_configs_path, get_scene_configs_path, join_path, load_yaml
from curobo._src.util.warp import init_warp


@dataclass
class _CollisionSettings:
    weight: torch.Tensor
    activation_distance: torch.Tensor


@dataclass
class _SelfCollisionSettings:
    pairs: torch.Tensor
    weight: torch.Tensor
    activation_distance: torch.Tensor


def _target_device_cfg(target: Union[DeviceCfg, torch.device, str], fallback: DeviceCfg) -> DeviceCfg:
    """Normalize a V2-style device target without making CUDA assumptions."""
    if isinstance(target, DeviceCfg):
        return target
    return DeviceCfg(torch.device(target), fallback.dtype, fallback.collision_geometry_dtype,
                     fallback.collision_gradient_dtype, fallback.collision_distance_dtype)


def _move_value(value: Any, target: DeviceCfg) -> Any:
    """Copy tensor-owned config values while retaining backend objects by identity."""
    if isinstance(value, torch.Tensor):
        return value.to(**target.as_torch_dict()).clone()
    if isinstance(value, DeviceCfg):
        return target
    if isinstance(value, list):
        return [_move_value(item, target) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, target) for item in value)
    if isinstance(value, dict):
        return {key: _move_value(item, target) for key, item in value.items()}
    move = getattr(value, "to", None)
    if callable(move) and is_dataclass(value):
        try:
            return move(target)
        except (TypeError, NotImplementedError):
            pass
    clone = getattr(value, "clone", None)
    if callable(clone) and is_dataclass(value):
        return clone()
    return value


def _copy_cost_config(config: Any, target: DeviceCfg, scene_checker: Any) -> Any:
    """Rebuild one cost config on ``target`` and reconnect its query resource."""
    values = {item.name: _move_value(getattr(config, item.name), target) for item in fields(config)}
    values["device_cfg"] = target
    if "_scene_collision_checker" in values:
        values["_scene_collision_checker"] = scene_checker
    return type(config)(**values)


def _clone_sampler(sampler: SampleBuffer, target: DeviceCfg) -> SampleBuffer:
    """Clone sampling state without advancing the deterministic sequence."""
    sequencer = deepcopy(sampler.sequencer)
    result = SampleBuffer(
        sequencer,
        sampler.ndims,
        target,
        sampler.up_bounds.to(**target.as_torch_dict()),
        sampler.low_bounds.to(**target.as_torch_dict()),
        store_buffer=None,
    )
    result._store_buffer = sampler._store_buffer
    result.fixed_samples = sampler.fixed_samples
    result._sample_buffer = (
        None
        if sampler._sample_buffer is None
        else sampler._sample_buffer.to(**target.as_torch_dict()).clone()
    )
    result._int_gen = torch.Generator(device=target.device)
    # CPU and MPS expose different generator-state encodings.  Preserve an
    # exact stream when their devices agree; cross-device moves restart the
    # target stream from the public sequencer seed rather than attempting an
    # invalid CUDA-style state transfer.
    if sampler.device_cfg.is_same_torch_device(target.device):
        result._int_gen.set_state(sampler._int_gen.get_state())
    else:
        result._int_gen.manual_seed(sampler.sequencer.seed)
    result._initial_state = sampler._initial_state.clone()
    return result


class _RobotSceneCollisionCfgPortableMixin:
    @property
    def num_envs(self) -> int:
        return self._num_envs_value

    def set_scene_collision_checker(self, *args, **kwargs):
        return self._set_scene_collision_checker(*args, **kwargs)

    def update_scene_model(self, *args, **kwargs):
        return self._update_scene_model(*args, **kwargs)

    def update_collision_parameters(self, *args, **kwargs):
        return self._update_collision_parameters(*args, **kwargs)

    def clone(self):
        return self._clone()

    def to(self, *args, **kwargs):
        return self._to(*args, **kwargs)


@dataclass
class RobotSceneCollisionCfg(_RobotSceneCollisionCfgPortableMixin):
    """Portable ownership record for a robot, its collision costs, and a world.

    The upstream record is constructed once around CUDA/Warp launch buffers.
    This implementation deliberately owns ordinary PyTorch value objects
    instead: :meth:`clone` and :meth:`to` make independent kinematics, cost,
    sampler, and built-in scene-query state.  A caller-provided checker is a
    live external resource and is therefore shared only on the same device;
    moving such a config requires the caller to supply a target-device
    checker via :meth:`set_scene_collision_checker`.
    """

    kinematics: Any
    sampler: Any
    bound_scale: torch.Tensor
    cspace_cost: Any
    self_collision_cost: Optional[Any] = None
    collision_cost: Optional[Any] = None
    collision_constraint: Optional[Any] = None
    scene_model: Optional[SceneCollision] = None
    rejection_ratio: int = 10
    device_cfg: DeviceCfg = DeviceCfg()
    contact_distance: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be DeviceCfg")
        if isinstance(self.rejection_ratio, bool) or not isinstance(self.rejection_ratio, int):
            raise TypeError("rejection_ratio must be an integer")
        if self.rejection_ratio < 1:
            raise ValueError("rejection_ratio must be positive")
        if not isinstance(self.contact_distance, (float, int)) or not isfinite(self.contact_distance):
            raise ValueError("contact_distance must be finite")
        if self.contact_distance < 0:
            raise ValueError("contact_distance must be non-negative")
        if not isinstance(self.bound_scale, torch.Tensor) or self.bound_scale.ndim not in (1, 2):
            raise ValueError("bound_scale must have shape [dof] or [1,dof]")
        if not self.device_cfg.is_same_torch_device(self.bound_scale.device):
            raise ValueError("bound_scale device does not match device_cfg")
        if self.bound_scale.shape[-1] != self.kinematics.dof:
            raise ValueError("bound_scale last dimension must match kinematics dof")
        for name in ("cspace_cost", "self_collision_cost", "collision_cost", "collision_constraint"):
            value = getattr(self, name)
            if value is not None and not self.device_cfg.is_same_torch_device(
                getattr(value, "device_cfg", self.device_cfg).device
            ):
                raise ValueError(f"{name} device does not match device_cfg")
        if self.scene_model is not None:
            self.set_scene_collision_checker(self.scene_model)

    @property
    def _num_envs_value(self) -> int:
        """Number of scene environments (one for a robot-only configuration)."""
        return 1 if self.scene_model is None else int(getattr(self.scene_model, "num_envs", 1))

    def _set_scene_collision_checker(self, scene_collision_checker: Optional[Any]) -> "RobotSceneCollisionCfg":
        """Attach or remove a collision checker and rewire both scene costs.

        A checker must expose the public sphere-query protocol.  Raw Warp
        objects are intentionally not accepted: callers should use
        :class:`SceneCollision`, which is backed by portable tensor queries.
        """
        if scene_collision_checker is not None:
            if not (hasattr(scene_collision_checker, "get_sphere_distance") or
                    hasattr(scene_collision_checker, "get_sphere_distance_raw")):
                raise TypeError("scene_collision_checker must expose a sphere-distance query")
            checker_cfg = getattr(scene_collision_checker, "device_cfg", None)
            if checker_cfg is not None and not self.device_cfg.is_same_torch_device(checker_cfg.device):
                raise ValueError("scene_collision_checker device does not match device_cfg")
        self.scene_model = scene_collision_checker
        for cost in (self.collision_cost, self.collision_constraint):
            if cost is None:
                continue
            cost.config.scene_collision_checker = scene_collision_checker
            if scene_collision_checker is not None:
                count = getattr(scene_collision_checker, "get_num_scene_collision_checkers", lambda: 1)()
                cost.config.update_num_scene_collision_checkers(count)
        return self

    def _update_scene_model(
        self,
        scene_model: Union[None, SceneCollision, SceneCfg, List[SceneCfg], Dict, List[Dict]],
        *,
        n_meshes: Optional[int] = None,
        n_cuboids: Optional[int] = None,
        max_collision_distance: float = 1.0,
    ) -> "RobotSceneCollisionCfg":
        """Replace the portable world while preserving cost object identity.

        Matching-environment :class:`SceneCollision` instances are updated in
        place, which lets a configured rollout retain its collision-cost
        buffers.  A changed environment count constructs a fresh portable
        checker and atomically reconnects the costs.
        """
        if scene_model is None:
            return self.set_scene_collision_checker(None)
        if isinstance(scene_model, SceneCollision):
            return self.set_scene_collision_checker(scene_model)
        if isinstance(scene_model, dict):
            scene_model = SceneCfg.create(scene_model)
        elif isinstance(scene_model, list):
            scene_model = [SceneCfg.create(item) if isinstance(item, dict) else item for item in scene_model]
            if not all(isinstance(item, SceneCfg) for item in scene_model):
                raise TypeError("scene_model list must contain SceneCfg or mappings")
        if not isinstance(scene_model, (SceneCfg, list)):
            raise TypeError("scene_model must be SceneCollision, SceneCfg, mapping, or a list of scenes")
        scenes = scene_model if isinstance(scene_model, list) else [scene_model]
        if not scenes:
            raise ValueError("scene_model list must not be empty")
        if max_collision_distance <= 0 or not isfinite(max_collision_distance):
            raise ValueError("max_collision_distance must be finite and positive")
        required_meshes = max(len(scene.mesh) for scene in scenes)
        required_cuboids = max(len(scene.cuboid) for scene in scenes)
        required_voxels = max(len(scene.voxel) for scene in scenes)
        if n_meshes is not None and n_meshes < required_meshes:
            raise ValueError("n_meshes is smaller than the supplied scene")
        if n_cuboids is not None and n_cuboids < required_cuboids:
            raise ValueError("n_cuboids is smaller than the supplied scene")
        can_update_in_place = (
            isinstance(self.scene_model, SceneCollision)
            and self.scene_model.num_envs == len(scenes)
            and required_meshes <= self.scene_model._world.mesh_cache.capacity
            and required_cuboids <= self.scene_model._world.primitive_cache.capacity
            and required_voxels <= self.scene_model._world.voxel_cache.capacity
        )
        if can_update_in_place:
            for env_idx, scene in enumerate(scenes):
                self.scene_model.load_collision_model(deepcopy(scene), env_idx)
            return self.set_scene_collision_checker(self.scene_model)
        checker = SceneCollision(SceneCollisionCfg(
            device_cfg=self.device_cfg,
            scene_model=deepcopy(scene_model),
            num_envs=len(scenes),
            max_distance=max_collision_distance,
            cache={
                "mesh": max(50, required_meshes) if n_meshes is None else n_meshes,
                "cuboid": max(50, required_cuboids) if n_cuboids is None else n_cuboids,
                "voxel": max(8, required_voxels),
            },
        ))
        return self.set_scene_collision_checker(checker)

    def _update_collision_parameters(
        self, collision_activation_distance: float, *, contact_distance: Optional[float] = None
    ) -> "RobotSceneCollisionCfg":
        """Update scene-cost activation and contact threshold in place."""
        if not isinstance(collision_activation_distance, (float, int)) or not isfinite(collision_activation_distance):
            raise ValueError("collision_activation_distance must be finite")
        if collision_activation_distance < 0:
            raise ValueError("collision_activation_distance must be non-negative")
        if contact_distance is None:
            contact_distance = 0.5 * float(collision_activation_distance)
        if not isfinite(contact_distance) or contact_distance < 0:
            raise ValueError("contact_distance must be finite and non-negative")
        activation = self.device_cfg.to_device([float(collision_activation_distance)])
        for cost in (self.collision_cost, self.collision_constraint):
            if cost is not None:
                cost.config.activation_distance = activation.clone()
        self.contact_distance = float(contact_distance)
        return self

    def _clone(self) -> "RobotSceneCollisionCfg":
        """Return an independent copy on the same device."""
        return self.to(self.device_cfg, copy=True)

    def _to(
        self, target: Union[DeviceCfg, torch.device, str], *, copy: bool = False
    ) -> "RobotSceneCollisionCfg":
        """Move portable state to CPU/MPS without reusing CUDA/Warp buffers."""
        target_cfg = _target_device_cfg(target, self.device_cfg)
        if target_cfg == self.device_cfg and not copy:
            return self

        source_cfg = self.kinematics.config
        kinematics_cfg = KinematicsCfg(
            target_cfg,
            list(source_cfg.tool_frames),
            source_cfg.kinematics_config.to(target_cfg),
            _move_value(source_cfg.self_collision_config, target_cfg),
            source_cfg.kinematics_parser,
            source_cfg.generator_config,
        )
        kinematics = Kinematics(
            kinematics_cfg,
            compute_jacobian=self.kinematics.compute_jacobian,
            compute_spheres=self.kinematics.compute_spheres,
            compute_com=self.kinematics.compute_com,
        )

        checker = self.scene_model
        if isinstance(checker, SceneCollision):
            model = deepcopy(checker.scene_model)
            checker = SceneCollision(SceneCollisionCfg(
                device_cfg=target_cfg,
                scene_model=model,
                num_envs=checker.num_envs,
                max_distance=1.0,
                cache={
                    "cuboid": checker._world.primitive_cache.capacity,
                    "mesh": checker._world.mesh_cache.capacity,
                    "voxel": checker._world.voxel_cache.capacity,
                },
            ))
        elif checker is not None:
            checker_cfg = getattr(checker, "device_cfg", None)
            if checker_cfg is not None and not target_cfg.is_same_torch_device(checker_cfg.device):
                raise ValueError(
                    "cannot move a caller-provided scene_collision_checker; "
                    "provide a checker on the target device"
                )

        cspace_cost = type(self.cspace_cost)(_copy_cost_config(self.cspace_cost.config, target_cfg, checker))
        self_collision_cost = None if self.self_collision_cost is None else type(self.self_collision_cost)(
            _copy_cost_config(self.self_collision_cost.config, target_cfg, checker)
        )
        collision_cost = None if self.collision_cost is None else type(self.collision_cost)(
            _copy_cost_config(self.collision_cost.config, target_cfg, checker)
        )
        collision_constraint = None if self.collision_constraint is None else type(self.collision_constraint)(
            _copy_cost_config(self.collision_constraint.config, target_cfg, checker)
        )
        return type(self)(
            kinematics=kinematics,
            sampler=_clone_sampler(self.sampler, target_cfg),
            bound_scale=self.bound_scale.to(**target_cfg.as_torch_dict()).clone(),
            cspace_cost=cspace_cost,
            self_collision_cost=self_collision_cost,
            collision_cost=collision_cost,
            collision_constraint=collision_constraint,
            scene_model=checker,
            rejection_ratio=self.rejection_ratio,
            device_cfg=target_cfg,
            contact_distance=self.contact_distance,
        )

    @staticmethod
    def load_from_config(
        robot_config: Union[RobotCfg, str] = "franka.yml",
        scene_model: Union[None, str, Dict, SceneCfg, List[SceneCfg], List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_envs: int = 1,
        n_meshes: int = 50,
        n_cuboids: int = 50,
        collision_activation_distance: float = 0.2,
        self_collision_activation_distance: float = 0.0,
        max_collision_distance: float = 1.0,
        scene_collision_checker: Optional[SceneCollision] = None,
        pose_weight: List[float] = [1, 1, 1, 1],
    ) -> "RobotSceneCollisionCfg":
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if n_meshes < 0 or n_cuboids < 0:
            raise ValueError("collision cache capacities must be nonnegative")
        if collision_activation_distance < 0 or self_collision_activation_distance < 0:
            raise ValueError("collision activation distances must be nonnegative")
        if max_collision_distance <= 0:
            raise ValueError("max_collision_distance must be positive")

        if isinstance(robot_config, KinematicsCfg):
            kin_cfg = robot_config
        elif hasattr(robot_config, "kinematics") and hasattr(
            robot_config.kinematics, "joint_names"
        ):
            # ``RobotCfg`` is a public input in V2.  Its kinematics record is
            # already the portable tree source, so retain it rather than
            # round-tripping through a YAML file.
            from curobo._src.robot.types import KinematicsParams

            raw = robot_config.kinematics
            kin_cfg = KinematicsCfg(
                device_cfg,
                list(raw.tool_frames),
                KinematicsParams(raw),
                self_collision_config=getattr(raw, "self_collision_config", None),
            )
        else:
            kin_cfg = KinematicsCfg.from_robot_yaml_file(robot_config, device_cfg=device_cfg)
        kinematics = Kinematics(kin_cfg, compute_spheres=True)
        # Collision scene environments route not just world geometry but also
        # the kinematics sphere buffer.  Start each environment from the same
        # reference spheres; attachments may subsequently mutate one row.
        parameters = kin_cfg.kinematics_config
        if num_envs > parameters.link_spheres.shape[0]:
            parameters._link_spheres = parameters.link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()
            parameters.reference_link_spheres = parameters.reference_link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()

        def _scene(value: Any) -> SceneCfg:
            if isinstance(value, SceneCfg):
                return value
            if isinstance(value, dict):
                return SceneCfg.create(value)
            if isinstance(value, str):
                from curobo.util_file import get_scene_configs_path, join_path, load_yaml

                return SceneCfg.create(load_yaml(join_path(get_scene_configs_path(), value)))
            raise TypeError(f"unsupported scene model: {type(value).__name__}")

        scene_values = None
        if scene_model is not None:
            scene_values = [_scene(x) for x in scene_model] if isinstance(scene_model, list) else _scene(scene_model)
        checker = scene_collision_checker
        if checker is None and scene_values is not None:
            checker = SceneCollision(SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_values,
                num_envs=num_envs,
                max_distance=max_collision_distance,
                cache={"cuboid": n_cuboids, "mesh": n_meshes, "voxel": 8},
            ))

        robot = kin_cfg.kinematics_config.robot_cfg
        spheres = list(robot.collision_spheres)
        ignored = {
            frozenset((a, b))
            for a, values in robot.self_collision_ignore.items()
            for b in values
        }
        pairs = [
            (i, j)
            for i in range(len(spheres))
            for j in range(i + 1, len(spheres))
            if spheres[i].link_name != spheres[j].link_name
            and frozenset((spheres[i].link_name, spheres[j].link_name)) not in ignored
        ]
        pair_tensor = torch.tensor(
            pairs, device=device_cfg.device, dtype=torch.long
        ).reshape(-1, 2)
        scalar = lambda value: torch.tensor(value, device=device_cfg.device, dtype=device_cfg.dtype)
        limits = kinematics.get_joint_limits()
        cspace_cfg = CSpaceCostCfg(
            weight=[1.0, 0.0],
            device_cfg=device_cfg,
            use_grad_input=True,
            cost_type=CSpaceCostType.POSITION,
            activation_distance=[0.0, 0.0],
            dof=kinematics.dof,
        )
        cspace_cfg.set_bounds(limits, teleport_mode=True)
        cspace_cost = PositionCSpaceCost(cspace_cfg)
        self_cfg = SelfCollisionKinematicsCfg(
            num_spheres=kinematics.total_spheres,
            collision_pairs=pair_tensor,
        )
        self_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=scalar(1.0),
                device_cfg=device_cfg,
                use_grad_input=True,
                self_collision_kin_config=self_cfg,
            )
        )
        collision_cost = collision_constraint = None
        if checker is not None:
            collision_cost = SceneCollisionCost(
                SceneCollisionCostCfg(
                    weight=scalar(1.0),
                    device_cfg=device_cfg,
                    use_grad_input=False,
                    activation_distance=collision_activation_distance,
                    num_spheres=kinematics.total_spheres,
                    sum_distance=False,
                    _scene_collision_checker=checker,
                )
            )
            collision_constraint = SceneCollisionCost(
                SceneCollisionCostCfg(
                    weight=scalar(1.0),
                    device_cfg=device_cfg,
                    use_grad_input=True,
                    activation_distance=0.0,
                    num_spheres=kinematics.total_spheres,
                    sum_distance=False,
                    _scene_collision_checker=checker,
                )
            )
        sampler = SampleBuffer.create_halton_sample_buffer(
            ndims=kinematics.dof,
            up_bounds=limits.position_upper_limits,
            low_bounds=limits.position_lower_limits,
            store_buffer=2000,
            seed=123,
            device_cfg=device_cfg,
        )
        return RobotSceneCollisionCfg(
            kinematics=kinematics,
            sampler=sampler,
            bound_scale=torch.ones(kinematics.dof, device=device_cfg.device, dtype=device_cfg.dtype),
            cspace_cost=cspace_cost,
            self_collision_cost=self_cost,
            collision_cost=collision_cost,
            collision_constraint=collision_constraint,
            scene_model=checker,
            device_cfg=device_cfg,
            # This is the pinned V2 threshold formula simplified for the
            # scalar portable activation distance.
            contact_distance=0.5 * float(collision_activation_distance),
        )
