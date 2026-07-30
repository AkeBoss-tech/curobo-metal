"""Pinned robot-scene collision object.

The low-level world and self-collision paths are production-backed. Joint-space
queries require a compatible kinematics object supplied in the configuration.
"""

from __future__ import annotations

import torch

from curobo_metal.ops.collision import sphere_sphere_signed_distance

from .collision_robot_scene_cfg import RobotSceneCollisionCfg


class RobotSceneCollision(RobotSceneCollisionCfg):
    def __init__(self, config: RobotSceneCollisionCfg) -> None:
        if not isinstance(config, RobotSceneCollisionCfg):
            raise TypeError("config must be a RobotSceneCollisionCfg")
        self.__dict__.update(config.__dict__)

    def get_kinematics(self, joint_position: torch.Tensor):
        if not callable(self.kinematics):
            raise TypeError("kinematics must be callable")
        return self.kinematics(joint_position)

    def get_collision_distance(self, x_sph, collision_buffer, weight, activation_distance,
                               env_query_idx=None, return_loss=False):
        if self.scene_model is None:
            return x_sph.new_zeros(x_sph.shape[:-1])
        return self.scene_model.get_sphere_distance_raw(
            x_sph, collision_buffer, weight, activation_distance, env_query_idx, return_loss
        )

    def get_self_collision_distance(self, x_sph: torch.Tensor) -> torch.Tensor:
        cost = self.self_collision_cost
        pairs = getattr(cost, "pairs", None)
        if pairs is None:
            raise NotImplementedError("self_collision_cost must expose a pairs tensor")
        result = sphere_sphere_signed_distance(x_sph, pairs)
        return (-result.reduced_distance).clamp_min(0)

    def clear_scene_cache(self) -> None:
        if self.scene_model is not None:
            self.scene_model.clear_cache()

    def update_world(self, scene_cfg) -> None:
        if self.scene_model is None:
            raise RuntimeError("scene_model is not configured")
        self.scene_model.load_collision_model(scene_cfg)

    @property
    def tool_frames(self):
        return getattr(self.kinematics, "tool_frames", None)

