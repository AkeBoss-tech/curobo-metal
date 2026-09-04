"""Pinned robot-scene collision object.

The low-level world and self-collision paths are production-backed. Joint-space
queries require a compatible kinematics object supplied in the configuration.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from curobo_metal.ops.collision import sphere_sphere_signed_distance

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float16_tensors, check_float32_tensors
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator

from .collision_robot_scene_cfg import RobotSceneCollisionCfg


class RobotSceneCollision(RobotSceneCollisionCfg):
    def __init__(self, config: RobotSceneCollisionCfg) -> None:
        if not isinstance(config, RobotSceneCollisionCfg):
            raise TypeError("config must be a RobotSceneCollisionCfg")
        self.__dict__.update(config.__dict__)
        self._collision_buffer = None

    def __getattr__(self, name):
        """Retain portable extensions without widening the pinned class AST.

        ``collision_buffer`` and ``get_sphere_distance`` predate this pinned
        module's smaller public declaration.  They remain available to the
        portable cost stack, but are resolved dynamically so the declared V2
        callable surface stays exact.
        """
        if name == "collision_buffer":
            return self._collision_buffer
        if name == "get_sphere_distance":
            return self._get_sphere_distance
        raise AttributeError(name)

    @property
    def tool_frames(self):
        """Configured end-effector link names.

        This is intentionally a live view of the portable kinematics model,
        matching the pinned cuRobo convenience property.  Robot reduction or
        a configuration reload can therefore change the value without a
        second collision-checker construction.
        """
        return self.kinematics.tool_frames

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        if self.cspace_cost is not None:
            self.cspace_cost.setup_batch_tensors(batch_size, horizon)
        if self.self_collision_cost is not None:
            self.self_collision_cost.setup_batch_tensors(batch_size, horizon)
        for cost in (self.collision_cost, self.collision_constraint):
            if cost is not None:
                cost.config.update_num_spheres(self.kinematics.total_spheres)
                cost.setup_batch_tensors(batch_size, horizon)
        self._collision_buffer = CollisionBuffer.from_shape(
            torch.Size((batch_size, horizon, self.kinematics.total_spheres, 4)),
            self.device_cfg,
        )

    def _get_kinematics(self, joint_position: torch.Tensor, idxs_env=None) -> KinematicsState:
        if not hasattr(self.kinematics, "_forward"):
            raise TypeError("kinematics must expose the pinned _forward interface")
        q = joint_position
        if q.ndim == 1:
            q = q[None, None]
        elif q.ndim == 2:
            q = q[:, None]
        return self.kinematics._forward(q, idxs_env)

    def get_kinematics(self, joint_position: torch.Tensor) -> KinematicsState:
        """Compute kinematics through the pinned public one-argument ABI."""
        if not isinstance(joint_position, torch.Tensor):
            raise TypeError("joint_position must be a torch.Tensor")
        if joint_position.ndim not in (1, 2, 3):
            raise ValueError(
                "joint_position must have shape [dof], [batch,dof], or [batch,horizon,dof]"
            )
        return self._get_kinematics(joint_position)

    def _buffer(self, spheres):
        if self._collision_buffer is None:
            self._collision_buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
        self._collision_buffer.resize(spheres.shape, self.device_cfg)
        return self._collision_buffer

    @staticmethod
    def _spheres(x_sph) -> torch.Tensor:
        """Normalize public sphere/state inputs and validate the portable ABI."""
        if isinstance(x_sph, KinematicsState):
            x_sph = x_sph.robot_spheres
        if not isinstance(x_sph, torch.Tensor):
            raise TypeError("x_sph must be a tensor or KinematicsState")
        if x_sph.ndim != 4 or x_sph.shape[-1] != 4:
            raise ValueError("x_sph must have shape [batch,horizon,spheres,4]")
        if not x_sph.is_floating_point():
            raise TypeError("x_sph must have a floating-point dtype")
        return x_sph

    def _zero_scene_distance(self, spheres: torch.Tensor) -> torch.Tensor:
        """Return a detached empty-world result and reset reusable diagnostics.

        The CUDA implementation owns launch buffers which are overwritten on
        every query.  This portable value implementation owns a reusable
        :class:`CollisionBuffer`; clearing it here prevents an earlier world
        query's gradient from leaking into a later robot-only query.
        """
        buffer = self._buffer(spheres)
        buffer.zero_()
        return spheres.new_zeros(spheres.shape[:-1])

    def get_collision_distance(
        self,
        x_sph: Union[torch.Tensor, KinematicsState],
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_sph = self._spheres(x_sph)
        if self.scene_model is None:
            return self._zero_scene_distance(x_sph)
        settings = self.collision_cost
        if settings is None:
            return self._zero_scene_distance(x_sph)
        if env_query_idx is not None and (
            not isinstance(env_query_idx, torch.Tensor)
            or
            env_query_idx.ndim != 1 or env_query_idx.shape[0] != x_sph.shape[0]
        ):
            raise ValueError("env_query_idx must have shape [batch]")
        return self.scene_model.get_sphere_distance_raw(
            x_sph, self._buffer(x_sph), settings.weight, settings.config.activation_distance,
            env_query_idx, False,
        )

    # SceneCollisionCost and older cuRobo consumers use this shorter spelling.
    # Keep it as a real forwarding API rather than a CUDA-only placeholder.
    def _get_sphere_distance(self, x_sph, env_query_idx=None):
        return self.get_collision_distance(x_sph, env_query_idx)

    def get_collision_constraint(
        self,
        x_sph: Union[torch.Tensor, KinematicsState],
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_sph = self._spheres(x_sph)
        if self.scene_model is None or self.collision_constraint is None:
            return self._zero_scene_distance(x_sph)
        value = self.scene_model.get_sphere_distance_raw(
            x_sph,
            self._buffer(x_sph),
            self.collision_constraint.weight,
            self.collision_constraint.config.activation_distance,
            env_query_idx,
            False,
        )
        return (-value).clamp_min(0)

    def get_self_collision_distance(self, x_sph: torch.Tensor) -> torch.Tensor:
        x_sph = self._spheres(x_sph)
        cost = self.self_collision_cost
        if cost is None:
            return x_sph.new_zeros((*x_sph.shape[:2], 1))
        pairs = getattr(getattr(cost, "config", None), "self_collision_kin_config", None)
        pairs = getattr(pairs, "collision_pairs", getattr(cost, "pairs", None))
        if pairs is None:
            return cost(x_sph).reshape(*x_sph.shape[:2], 1)
        if pairs.numel() == 0:
            return x_sph.new_zeros((*x_sph.shape[:2], 1))
        prefix = x_sph.shape[:-2]
        collision_spheres = x_sph
        if bool((x_sph[..., 3] < 0).any().item()):
            collision_spheres = x_sph.clone()
            collision_spheres[..., 3].clamp_min_(0)
        result = sphere_sphere_signed_distance(
            collision_spheres.reshape(-1, *x_sph.shape[-2:]),
            pairs.to(device=x_sph.device, dtype=torch.int64),
        )
        activation = getattr(getattr(cost, "config", None), "activation_distance", None)
        # Preserve the singleton aggregate-cost axis exposed by the pinned
        # CUDA implementation: world distance is sphere-resolved while self
        # distance has shape [batch, horizon, 1].
        value = (-result.reduced_distance).clamp_min(0)
        if activation is not None:
            value = (value - torch.as_tensor(activation, device=value.device, dtype=value.dtype)).clamp_min(0)
        return value.reshape(*prefix, 1)

    def get_self_collision(self, x_sph: torch.Tensor) -> torch.Tensor:
        # The public one-shot query exposes one aggregate value per
        # batch/horizon entry; the lower-level distance API retains the final
        # singleton cost axis for rollout composition.
        return self.get_self_collision_distance(x_sph).squeeze(-1)

    def get_collision_vector(
        self,
        x_sph: Union[torch.Tensor, KinematicsState],
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sph = self._spheres(x_sph)
        distance = self.get_collision_distance(x_sph, env_query_idx)
        return distance.detach(), self._buffer(x_sph).gradient.clone()

    def get_scene_self_collision_distance_from_joints(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(q, torch.Tensor) or q.ndim != 3:
            raise ValueError("q must have shape [batch,horizon,dof]")
        state = self._get_kinematics(q, env_query_idx)
        return (
            self.get_collision_distance(state, env_query_idx),
            self.get_self_collision_distance(state),
        )

    def get_scene_self_collision_distance_from_joint_trajectory(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_scene_self_collision_distance_from_joints(q, env_query_idx)

    def get_bound(
        self, q: torch.Tensor, q_tau: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q_tau is not None and q_tau.shape != q.shape:
            raise ValueError("q_tau must have the same shape as q")
        limits = self.kinematics.get_joint_limits()
        lower, upper = self._position_bounds(limits, q)
        return (lower - q).clamp_min(0) + (q - upper).clamp_min(0)

    def _position_bounds(self, limits, reference):
        """Accept both V2 ``JointLimits`` tensors and legacy named mappings."""
        if hasattr(limits, "position_lower_limits"):
            return (
                limits.position_lower_limits.to(device=reference.device, dtype=reference.dtype),
                limits.position_upper_limits.to(device=reference.device, dtype=reference.dtype),
            )
        lower = reference.new_tensor([limits[n].lower for n in self.kinematics.joint_names])
        upper = reference.new_tensor([limits[n].upper for n in self.kinematics.joint_names])
        return lower, upper

    def sample(
        self,
        n: int,
        mask_valid: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(n, int) or n < 0:
            raise ValueError("sample_size must be a nonnegative integer")
        if n == 0:
            return torch.empty((0, self.kinematics.dof), **self.device_cfg.as_torch_dict())
        if self.sampler is not None:
            count = n if not mask_valid else n * self.rejection_ratio
            samples = self.sampler.get_samples(count, bounded=True)
            if not mask_valid:
                return samples
            accepted = samples[self.validate(samples, env_query_idx)]
            if len(accepted) >= n:
                return accepted[:n]
        limits = self.kinematics.get_joint_limits()
        reference = torch.empty((), **self.device_cfg.as_torch_dict())
        lower, upper = self._position_bounds(limits, reference)
        samples = lower + torch.rand(
            n, self.kinematics.dof, device=lower.device, dtype=lower.dtype
        ) * (upper - lower)
        if not mask_valid:
            return samples
        accepted = [samples[self.validate(samples, env_query_idx)]]
        for _ in range(self.rejection_ratio - 1):
            if sum(len(x) for x in accepted) >= n:
                break
            candidate = lower + torch.rand(
                n, self.kinematics.dof,
                device=lower.device, dtype=lower.dtype,
            ) * (upper - lower)
            accepted.append(candidate[self.validate(candidate, env_query_idx)])
        output = torch.cat(accepted)
        if len(output) < n:
            raise RuntimeError("unable to sample enough collision-free configurations")
        return output[:n]

    def validate(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not isinstance(q, torch.Tensor):
            raise TypeError("joint_position must be a torch.Tensor")
        if q.ndim not in (1, 2, 3):
            raise ValueError("joint_position must have shape [dof], [batch,dof], or [batch,horizon,dof]")
        if q.shape[-1] != self.kinematics.dof:
            raise ValueError("joint_position dof does not match kinematics")
        state = self._get_kinematics(q, env_query_idx)
        world = self.get_collision_distance(state, env_query_idx)
        self_distance = self.get_self_collision_distance(state)
        # Existing portable callers consume self collision as [B,H], while
        # the world query is sphere-resolved [B,H,S].  Normalize only for the
        # predicate so B/H axes never accidentally broadcast against S.
        if self_distance.ndim == world.ndim - 1:
            self_distance = self_distance.unsqueeze(-1)
        bound = self.get_bound(q)
        valid = (
            (world >= 0).all(dim=-1)
            & (self_distance <= 0).all(dim=-1)
        )
        if q.ndim == 2 and valid.ndim == 2:
            valid = valid[:, 0]
        return valid & (bound <= 0).all(dim=-1)

    def sample_trajectory(
        self,
        batch: int,
        horizon: int,
        mask_valid: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample = self.sample(batch * horizon, mask_valid=False).reshape(
            batch, horizon, -1
        )
        if not mask_valid:
            return sample
        valid = self.validate_trajectory(sample, env_query_idx)
        return sample[valid]

    def validate_trajectory(
        self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        valid = self.validate(q, env_query_idx)
        return valid.all(dim=-1) if valid.ndim > 1 else valid

    def get_active_js(self, full_js: JointState) -> JointState:
        return self.kinematics.get_active_js(full_js)

    def pose_distance(
        self, x_des: Pose, x_current: Pose, resize: bool = False
    ) -> torch.Tensor:
        del resize
        position = torch.linalg.vector_norm(x_des.position - x_current.position, dim=-1)
        dot = torch.abs((x_des.quaternion * x_current.quaternion).sum(dim=-1)).clamp(0, 1)
        rotation = 2.0 * torch.acos(dot)
        return position, rotation

    def get_point_robot_distance(self, points: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Return signed point-to-robot distance using the sphere envelope.

        Positive values indicate a point inside at least one robot sphere,
        as in pinned V2.  A single robot configuration broadcasts across a
        batched point cloud; otherwise robot and point-cloud batch counts
        must agree.  This path is composed PyTorch and remains differentiable
        with respect to both point positions and joint positions.
        """
        if not isinstance(points, torch.Tensor) or not isinstance(q, torch.Tensor):
            raise TypeError("points and q must be tensors")
        if points.shape[-1] != 3:
            raise ValueError("points must end in xyz")
        if points.ndim not in (2, 3):
            raise ValueError("points must have shape [points,3] or [batch,points,3]")
        if q.ndim != 2:
            raise ValueError("q must have shape [batch,dof]")
        if q.shape[-1] != self.kinematics.dof:
            raise ValueError("q dof does not match kinematics")
        if q.device != points.device:
            raise ValueError("points and q must share a device")
        if q.dtype != points.dtype:
            raise ValueError("points and q must share a dtype")
        spheres = self.get_kinematics(q).robot_spheres.squeeze(1)
        squeeze = points.ndim == 2
        query = points.unsqueeze(0) if squeeze else points
        if spheres.shape[0] not in (1, query.shape[0]):
            raise ValueError("robot batch must be one or match point-cloud batch")
        delta = query[:, :, None, :] - spheres[:, None, :, :3]
        penetration = spheres[:, None, :, 3] - torch.linalg.vector_norm(delta, dim=-1)
        result = penetration.amax(dim=-1)
        return result.squeeze(0) if squeeze else result

    def clear_scene_cache(self) -> None:
        if self.scene_model is not None:
            self.scene_model.clear_cache()

    def update_world(self, scene_cfg: SceneCfg) -> None:
        if self.scene_model is None:
            raise RuntimeError("scene_model is not configured")
        if isinstance(scene_cfg, (list, tuple)):
            if len(scene_cfg) != self.scene_model.num_envs:
                raise ValueError("scene_cfg list must contain one scene per environment")
            for environment, scene in enumerate(scene_cfg):
                self.scene_model.load_collision_model(scene, environment)
        else:
            self.scene_model.load_collision_model(scene_cfg)

    @property
    def tool_frames(self):
        return getattr(self.kinematics, "tool_frames", None)
