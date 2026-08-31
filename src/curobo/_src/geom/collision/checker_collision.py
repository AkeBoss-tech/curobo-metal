"""Pinned stateless collision-checker facade over portable scene queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.wp_autograd import (
    SphereObstacleCollision,
    SweptSphereObstacleCollision,
)
from curobo._src.types.device_cfg import DeviceCfg

if TYPE_CHECKING:
    from curobo._src.geom.data.data_scene import SceneData


@dataclass
class _CollisionCheckerPortable:
    device_cfg: DeviceCfg
    max_distance: Union[float, torch.Tensor] = 1.0
    _max_distance_t: Optional[torch.Tensor] = field(default=None, repr=False)
    _scene_cache: dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._max_distance_t = (
            self.device_cfg.to_device([self.max_distance])
            if isinstance(self.max_distance, float)
            else self.max_distance
        )

    def _checker(self, scene):
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        if isinstance(scene, SceneCollision):
            return scene
        model = getattr(scene, "scene_model", None)
        if model is None:
            raise NotImplementedError(
                "portable CollisionChecker requires SceneData.scene_model; raw Warp "
                "scene tensor descriptors are unavailable"
            )
        key = id(scene)
        if key not in self._scene_cache:
            self._scene_cache[key] = SceneCollision(SceneCollisionCfg(
                self.device_cfg, model, getattr(scene, "num_envs", 1),
                float(self._max_distance_t.reshape(-1)[0].item()),
            ))
        return self._scene_cache[key]

    def get_sphere_distance(
        self, scene, query_sphere, collision_buffer, weight, activation_distance,
        env_query_idx=None, return_loss=False,
    ):
        return self._checker(scene).get_sphere_distance_raw(
            query_sphere, collision_buffer, weight, activation_distance,
            env_query_idx, return_loss,
        )

    def get_swept_sphere_distance(
        self, scene, query_sphere, collision_buffer, weight, activation_distance,
        trajectory_dt, enable_speed_metric=False, env_query_idx=None, return_loss=False,
    ):
        return self._checker(scene).get_swept_sphere_distance_raw(
            query_sphere, collision_buffer, weight, activation_distance,
            trajectory_dt, enable_speed_metric, env_query_idx, return_loss,
        )

    def get_sphere_collision(
        self, scene, query_sphere, collision_buffer, weight, activation_distance,
        env_query_idx=None, return_loss=False,
    ):
        return self.get_sphere_distance(
            scene, query_sphere, collision_buffer, weight, activation_distance,
            env_query_idx, return_loss,
        )

    def get_swept_sphere_collision(
        self, scene, query_sphere, collision_buffer, weight, activation_distance,
        trajectory_dt, enable_speed_metric=False, env_query_idx=None, return_loss=False,
    ):
        return self.get_swept_sphere_distance(
            scene, query_sphere, collision_buffer, weight, activation_distance,
            trajectory_dt, enable_speed_metric, env_query_idx, return_loss,
        )


@dataclass
class CollisionChecker:
    """Pinned cuRoboV2 declaration surface for portable scene collision."""

    device_cfg: DeviceCfg
    max_distance: Union[float, torch.Tensor] = 1.0
    _max_distance_t: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        raise NotImplementedError

    def get_sphere_distance(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_swept_sphere_distance(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_sphere_collision(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_swept_sphere_collision(
        self,
        scene: "SceneData",
        query_sphere: torch.Tensor,
        collision_buffer: CollisionBuffer,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        trajectory_dt: torch.Tensor,
        enable_speed_metric: bool = False,
        env_query_idx: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


# Runtime remains the high-level SceneCollision-backed portable checker.  Raw
# Warp scene tensor descriptors stay explicit unsupported boundaries there.
if not TYPE_CHECKING:
    CollisionChecker = _CollisionCheckerPortable


__all__ = ["CollisionChecker"]
