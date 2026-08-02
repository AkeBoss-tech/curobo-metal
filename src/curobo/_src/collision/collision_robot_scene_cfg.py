"""Pinned robot-scene collision configuration record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
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


@dataclass
class _CollisionSettings:
    weight: torch.Tensor
    activation_distance: torch.Tensor


@dataclass
class _SelfCollisionSettings:
    pairs: torch.Tensor
    weight: torch.Tensor
    activation_distance: torch.Tensor


@dataclass
class RobotSceneCollisionCfg:
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

    @staticmethod
    def load_from_config(
        robot_config: Union[Any, str] = "franka.yml",
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
        kin_cfg = (
            robot_config
            if isinstance(robot_config, KinematicsCfg)
            else KinematicsCfg.from_robot_yaml_file(robot_config, device_cfg=device_cfg)
        )
        kinematics = Kinematics(kin_cfg, compute_spheres=True)

        def _scene(value: Any) -> SceneCfg:
            if isinstance(value, SceneCfg):
                return value
            if isinstance(value, dict):
                return SceneCfg.create(value)
            if isinstance(value, str):
                from curobo.util_file import get_world_configs_path, join_path, load_yaml
                return SceneCfg.create(load_yaml(join_path(get_world_configs_path(), value)))
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
        return RobotSceneCollisionCfg(
            kinematics=kinematics,
            sampler=None,
            bound_scale=torch.ones(kinematics.dof, device=device_cfg.device, dtype=device_cfg.dtype),
            cspace_cost=None,
            self_collision_cost=_SelfCollisionSettings(
                pair_tensor, scalar(1.0), scalar(self_collision_activation_distance)
            ),
            collision_cost=_CollisionSettings(
                scalar(1.0), scalar(collision_activation_distance)
            ),
            collision_constraint=_CollisionSettings(
                scalar(1.0), scalar(collision_activation_distance)
            ),
            scene_model=checker,
            device_cfg=device_cfg,
            contact_distance=float(collision_activation_distance),
        )
