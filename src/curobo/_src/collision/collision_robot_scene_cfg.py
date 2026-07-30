"""Pinned robot-scene collision configuration record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.types import SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


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
        raise NotImplementedError(
            "robot-config construction depends on the pinned kinematics/cost namespace; "
            "construct RobotSceneCollisionCfg from production components"
        )
