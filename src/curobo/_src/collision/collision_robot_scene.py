"""Pinned robot-scene collision object.

The low-level world and self-collision paths are production-backed. Joint-space
queries require a compatible kinematics object supplied in the configuration.
"""

from __future__ import annotations

import torch

from curobo_metal.ops.collision import sphere_sphere_signed_distance

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose

from .collision_robot_scene_cfg import RobotSceneCollisionCfg


class RobotSceneCollision(RobotSceneCollisionCfg):
    def __init__(self, config: RobotSceneCollisionCfg) -> None:
        if not isinstance(config, RobotSceneCollisionCfg):
            raise TypeError("config must be a RobotSceneCollisionCfg")
        self.__dict__.update(config.__dict__)
        self._collision_buffer = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        self._collision_buffer = CollisionBuffer.from_shape(
            torch.Size((batch_size, horizon, self.kinematics.total_spheres, 4)),
            self.device_cfg,
        )

    def get_kinematics(self, joint_position: torch.Tensor, idxs_env=None):
        if not hasattr(self.kinematics, "_forward"):
            raise TypeError("kinematics must expose the pinned _forward interface")
        q = joint_position
        if q.ndim == 1:
            q = q[None, None]
        elif q.ndim == 2:
            q = q[:, None]
        return self.kinematics._forward(q, idxs_env)

    def _buffer(self, spheres):
        if self._collision_buffer is None:
            self._collision_buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
        self._collision_buffer.resize(spheres.shape, self.device_cfg)
        return self._collision_buffer

    def get_collision_distance(self, x_sph, env_query_idx=None):
        if isinstance(x_sph, KinematicsState):
            x_sph = x_sph.robot_spheres
        if self.scene_model is None:
            return x_sph.new_zeros(x_sph.shape[:-1])
        settings = self.collision_cost
        return self.scene_model.get_sphere_distance_raw(
            x_sph, self._buffer(x_sph), settings.weight, settings.activation_distance,
            env_query_idx, False,
        )

    # SceneCollisionCost and older cuRobo consumers use this shorter spelling.
    # Keep it as a real forwarding API rather than a CUDA-only placeholder.
    def get_sphere_distance(self, x_sph, env_query_idx=None):
        return self.get_collision_distance(x_sph, env_query_idx)

    @property
    def collision_buffer(self):
        return self._collision_buffer

    def get_collision_constraint(self, x_sph, env_query_idx=None):
        value = self.get_collision_distance(x_sph, env_query_idx)
        return (-value).clamp_min(0)

    def get_self_collision_distance(self, x_sph) -> torch.Tensor:
        if isinstance(x_sph, KinematicsState):
            x_sph = x_sph.robot_spheres
        cost = self.self_collision_cost
        if cost is None:
            return x_sph.new_zeros((*x_sph.shape[:2], 1))
        pairs = getattr(cost, "pairs", None)
        if pairs is None:
            raise NotImplementedError("self_collision_cost must expose a pairs tensor")
        prefix = x_sph.shape[:-2]
        result = sphere_sphere_signed_distance(x_sph.reshape(-1, *x_sph.shape[-2:]), pairs)
        return (-result.reduced_distance).clamp_min(0).reshape(prefix)

    def get_self_collision(self, x_sph):
        return self.get_self_collision_distance(x_sph)

    def get_collision_vector(self, x_sph, env_query_idx=None):
        if isinstance(x_sph, KinematicsState):
            x_sph = x_sph.robot_spheres
        distance = self.get_collision_distance(x_sph, env_query_idx)
        return distance.detach(), self._buffer(x_sph).gradient.clone()

    def get_scene_self_collision_distance_from_joints(self, q, env_query_idx=None):
        state = self.get_kinematics(q, env_query_idx)
        return (
            self.get_collision_distance(state, env_query_idx),
            self.get_self_collision_distance(state),
        )

    def get_scene_self_collision_distance_from_joint_trajectory(
        self, q, env_query_idx=None
    ):
        return self.get_scene_self_collision_distance_from_joints(q, env_query_idx)

    def get_bound(self, joint_position, q_tau=None):
        if q_tau is not None and q_tau.shape != joint_position.shape:
            raise ValueError("q_tau must have the same shape as q")
        limits = self.kinematics.get_joint_limits()
        lower = joint_position.new_tensor([limits[n].lower for n in self.kinematics.joint_names])
        upper = joint_position.new_tensor([limits[n].upper for n in self.kinematics.joint_names])
        return (lower - joint_position).clamp_min(0) + (joint_position - upper).clamp_min(0)

    def sample(self, sample_size, mask_valid=True, env_query_idx=None):
        limits = self.kinematics.get_joint_limits()
        lower = self.device_cfg.to_device([limits[n].lower for n in self.kinematics.joint_names])
        upper = self.device_cfg.to_device([limits[n].upper for n in self.kinematics.joint_names])
        samples = lower + torch.rand(
            sample_size, self.kinematics.dof, device=lower.device, dtype=lower.dtype
        ) * (upper - lower)
        if not mask_valid:
            return samples
        accepted = [samples[self.validate(samples, env_query_idx)]]
        for _ in range(self.rejection_ratio - 1):
            if sum(len(x) for x in accepted) >= sample_size:
                break
            candidate = lower + torch.rand(
                sample_size, self.kinematics.dof,
                device=lower.device, dtype=lower.dtype,
            ) * (upper - lower)
            accepted.append(candidate[self.validate(candidate, env_query_idx)])
        output = torch.cat(accepted)
        if len(output) < sample_size:
            raise RuntimeError("unable to sample enough collision-free configurations")
        return output[:sample_size]

    def validate(self, joint_position, env_query_idx=None):
        state = self.get_kinematics(joint_position, env_query_idx)
        world = self.get_collision_distance(state, env_query_idx)
        self_distance = self.get_self_collision_distance(state)
        bound = self.get_bound(joint_position)
        valid = (
            (world >= 0).all(dim=-1)
            & (self_distance <= 0)
        )
        if joint_position.ndim == 2 and valid.ndim == 2:
            valid = valid[:, 0]
        return valid & (bound <= 0).all(dim=-1)

    def sample_trajectory(self, sample_size, horizon, mask_valid=True, env_query_idx=None):
        sample = self.sample(sample_size * horizon, mask_valid=False).reshape(
            sample_size, horizon, -1
        )
        if not mask_valid:
            return sample
        valid = self.validate_trajectory(sample, env_query_idx)
        return sample[valid]

    def validate_trajectory(self, joint_position, env_query_idx=None):
        valid = self.validate(joint_position, env_query_idx)
        return valid.all(dim=-1) if valid.ndim > 1 else valid

    def get_active_js(self, joint_state: JointState):
        return self.kinematics.get_active_js(joint_state)

    def pose_distance(self, x_des: Pose, x_current: Pose, resize: bool = False):
        del resize
        position = torch.linalg.vector_norm(x_des.position - x_current.position, dim=-1)
        dot = torch.abs((x_des.quaternion * x_current.quaternion).sum(dim=-1)).clamp(0, 1)
        rotation = 2.0 * torch.acos(dot)
        return position, rotation

    def get_point_robot_distance(self, points: torch.Tensor, q: torch.Tensor):
        if points.shape[-1] != 3:
            raise ValueError("points must end in xyz")
        spheres = self.get_kinematics(q).robot_spheres
        delta = points[..., None, :] - spheres[..., None, :, :3]
        return (
            torch.linalg.vector_norm(delta, dim=-1) - spheres[..., None, :, 3]
        ).min(dim=-1).values

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
