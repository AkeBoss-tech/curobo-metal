"""Portable obstacle attachment management for collision-aware planning."""

from __future__ import annotations

from typing import List, Optional

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitResult, SphereFitType
from curobo._src.geom.types import Capsule, Cuboid, Cylinder, Mesh, Obstacle, Sphere
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_info


class _VertexMesh:
    def __init__(self, vertices: torch.Tensor):
        self.vertices = vertices


def _primitive_vertices(obstacle: Obstacle, device_cfg: DeviceCfg) -> torch.Tensor:
    dtype, device = device_cfg.dtype, device_cfg.device
    if isinstance(obstacle, Mesh):
        vertices = torch.as_tensor(obstacle.vertices, device=device, dtype=dtype)
    elif isinstance(obstacle, Cuboid):
        half = torch.tensor(obstacle.dims, device=device, dtype=dtype) * 0.5
        signs = torch.tensor(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            device=device, dtype=dtype,
        )
        vertices = signs * half
    elif isinstance(obstacle, Sphere):
        axis = torch.eye(3, device=device, dtype=dtype)
        vertices = torch.cat((axis, -axis), dim=0) * obstacle.radius
    elif isinstance(obstacle, Capsule):
        base = torch.tensor(obstacle.base, device=device, dtype=dtype)
        tip = torch.tensor(obstacle.tip, device=device, dtype=dtype)
        axis = torch.eye(3, device=device, dtype=dtype) * obstacle.radius
        vertices = torch.cat((base + axis, base - axis, tip + axis, tip - axis), dim=0)
    elif isinstance(obstacle, Cylinder):
        angle = torch.arange(16, device=device, dtype=dtype) * (2 * torch.pi / 16)
        ring = torch.stack(
            (obstacle.radius * angle.cos(), obstacle.radius * angle.sin()), dim=-1
        )
        low = torch.cat(
            (ring, torch.full((16, 1), -0.5 * obstacle.height, device=device, dtype=dtype)),
            dim=-1,
        )
        high = low.clone()
        high[:, 2] *= -1
        vertices = torch.cat((low, high), dim=0)
    else:
        raise NotImplementedError(
            f"attachment sphere fitting does not support {type(obstacle).__name__}"
        )
    if obstacle.pose is not None:
        pose = Pose.from_list(obstacle.pose, device_cfg)
        vertices = pose.transform_points(vertices).reshape(-1, 3)
    return vertices


class _AttachmentManagerPortableMixin:
    @property
    def attached_link_name(self) -> Optional[str]:
        return self._attached_link_name

    @property
    def last_fit_result(self) -> Optional[SphereFitResult]:
        return self._last_fit_result


class AttachmentManager(_AttachmentManagerPortableMixin):
    """Own the single portable object attachment associated with a robot model.

    The pinned V2 manager owns one attachment at a time.  This implementation
    retains that ownership model, but makes the mutation transactional across
    environment banks: a failed validation cannot resize the robot spheres or
    partially disable a world object.  The actual sphere fitting and FK remain
    ordinary PyTorch operations and therefore work on CPU and MPS.
    """
    def __init__(
        self,
        kinematics: Kinematics,
        scene_collision: Optional[SceneCollision] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        if not isinstance(kinematics, Kinematics):
            raise TypeError("kinematics must be Kinematics")
        self._kinematics = kinematics
        self._scene_collision = scene_collision
        self._device_cfg = device_cfg
        self._last_fit_result: Optional[SphereFitResult] = None
        self._attached_link_name: Optional[str] = None
        self._disabled_obstacle_names: List[str] = []
        self._disabled_num_envs = 0

    @property
    def kinematics_params(self) -> KinematicsParams:
        return self._kinematics.config.kinematics_config

    @property
    def _attached_link_name_value(self) -> Optional[str]:
        """Name of the currently attached link, or ``None`` when detached.

        This is an additive portable lifecycle inspection helper.  Callers
        needing the exact V2 private field can continue using
        ``_attached_link_name``.
        """
        return self._attached_link_name

    @property
    def _last_fit_result_value(self) -> Optional[SphereFitResult]:
        """Most recent deterministic sphere-fit result, if any."""
        return self._last_fit_result

    def fit_spheres(
        self,
        obstacles: List[Obstacle],
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
    ) -> torch.Tensor:
        if not obstacles:
            raise ValueError("obstacles must be non-empty")
        if not all(isinstance(value, Obstacle) for value in obstacles):
            raise TypeError("obstacles must contain Obstacle values")
        mesh = _VertexMesh(
            torch.cat(
                [_primitive_vertices(value, self._device_cfg) for value in obstacles],
                dim=0,
            )
        )
        result = fit_spheres_to_mesh(
            mesh,
            num_spheres=num_spheres,
            surface_radius=surface_radius,
            fit_type=sphere_fit_type,
            device_cfg=self._device_cfg,
        )
        self._last_fit_result = result
        fit_name = sphere_fit_type.value if isinstance(sphere_fit_type, SphereFitType) else sphere_fit_type
        log_info(
            f"AttachmentManager.fit_spheres: fitted {result.num_spheres} spheres "
            f"using portable {fit_name}"
        )
        return torch.cat((result.centers, result.radii[:, None]), dim=-1)

    def _validate_joint_states(self, joint_states: JointState) -> torch.Tensor:
        if not isinstance(joint_states, JointState):
            raise TypeError("joint_states must be JointState")
        q = joint_states.position
        if not isinstance(q, torch.Tensor) or q.ndim not in (1, 2):
            raise ValueError("joint_states.position must have shape [dof] or [env, dof]")
        if q.shape[-1] != self._kinematics.dof:
            raise ValueError(
                f"joint_states has dof={q.shape[-1]}, expected {self._kinematics.dof}"
            )
        q = q.to(device=self._device_cfg.device, dtype=self._device_cfg.dtype)
        if not bool(torch.isfinite(q).all().item()):
            raise ValueError("joint_states.position must contain finite values")
        return q.unsqueeze(0) if q.ndim == 1 else q

    def _object_to_link_poses(
        self,
        q: torch.Tensor,
        world_objects_pose_offset: Optional[Pose],
    ) -> Optional[Pose]:
        """Resolve an object pose once and normalize its environment rank."""
        if world_objects_pose_offset is None:
            return None
        if not isinstance(world_objects_pose_offset, Pose):
            raise TypeError("world_objects_pose_offset must be a Pose")
        offset = world_objects_pose_offset.clone().to(device=self._device_cfg.device)
        if offset.position is None or offset.quaternion is None:
            raise ValueError("world_objects_pose_offset must contain position and quaternion")
        if offset.position.dtype != q.dtype:
            offset = Pose(offset.position.to(dtype=q.dtype), offset.quaternion.to(dtype=q.dtype))
        pose_count = offset.position.reshape(-1, 3).shape[0]
        if pose_count not in (1, q.shape[0]):
            raise ValueError(
                "world_objects_pose_offset must have one pose or one pose per environment"
            )
        if pose_count == 1 and q.shape[0] > 1:
            offset = Pose(
                offset.position.reshape(1, 3).expand(q.shape[0], -1),
                offset.quaternion.reshape(1, 4).expand(q.shape[0], -1),
            )
        state = self._kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self._kinematics.joint_names)
        )
        if state.tool_poses is None:
            log_and_raise("FK result has no tool_poses; cannot resolve attachment offset.")
        link_pose = state.tool_poses.get_link_pose(self._kinematics.tool_frames[0])
        return link_pose.inverse().multiply(offset)

    def _validate_disable_names(self, names: List[str], num_envs: int) -> None:
        """Validate every target before changing any scene enable bit."""
        if self._scene_collision is None:
            return
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError("disable_obstacle_names must contain non-empty names")
            for environment in range(num_envs):
                if not self._scene_collision.check_obstacle_exists(name, environment):
                    raise ValueError(
                        f"obstacle {name!r} does not exist in environment {environment}"
                    )

    def update(
        self,
        sphere_tensor: torch.Tensor,
        joint_states: JointState,
        link_name: str = "attached_object",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        q = self._validate_joint_states(joint_states)
        values = torch.as_tensor(
            sphere_tensor,
            device=self._device_cfg.device,
            dtype=self._device_cfg.dtype,
        )
        if values.ndim != 2 or values.shape[-1] != 4:
            raise ValueError("sphere_tensor must have shape [num_spheres,4]")
        if values.shape[0] == 0:
            raise ValueError("sphere_tensor must contain at least one sphere")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("sphere_tensor must contain finite values")
        num_envs = q.shape[0]
        params = self.kinematics_params
        indices = params.get_sphere_index_from_link_name(link_name)
        if not len(indices):
            raise ValueError(f"link '{link_name}' has no allocated sphere slots")
        if len(values) > len(indices):
            raise ValueError(
                f"fitted {len(values)} spheres but link '{link_name}' has only "
                f"{len(indices)} sphere slots"
            )
        object_to_link = self._object_to_link_poses(q, world_objects_pose_offset)
        # ``set_num_envs`` extends from immutable reference spheres instead of
        # duplicating a currently attached environment.  That makes an update
        # from one grasp to many deterministic and keeps detach reversible.
        # Resolve/validate the optional offset first so a malformed pose rank
        # cannot resize this caller-owned configuration bank.
        params.set_num_envs(num_envs)
        padding = values.new_zeros((len(indices), 4))
        padding[:, 3] = -100.0
        for environment in range(num_envs):
            current = values
            if object_to_link is not None:
                pose = Pose(object_to_link.position[environment : environment + 1],
                            object_to_link.quaternion[environment : environment + 1])
                centers = pose.transform_points(values[:, :3]).reshape(-1, 3)
                current = torch.cat((centers, values[:, 3:]), dim=-1)
            padding[: len(current)] = current
            params.link_spheres[environment, indices] = padding
        self._attached_link_name = link_name

    def attach(
        self,
        joint_states: JointState,
        obstacles: List[Obstacle],
        link_name: str = "attached_object",
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
        world_objects_pose_offset: Optional[Pose] = None,
        disable_obstacle_names: Optional[List[str]] = None,
    ) -> None:
        q = self._validate_joint_states(joint_states)
        names = list(disable_obstacle_names or [])
        self._validate_disable_names(names, q.shape[0])
        # The upstream record has one attached-link field.  Do not leave a
        # prior attachment or its disabled world geometry stranded when this
        # manager is reused for a new payload.
        if self._attached_link_name is not None or self._disabled_obstacle_names:
            self.detach()
        values = self.fit_spheres(
            obstacles, num_spheres, surface_radius, sphere_fit_type
        )
        self.update(values, JointState.from_position(q, joint_names=joint_states.joint_names), link_name, world_objects_pose_offset)
        if names and self._scene_collision is not None:
            for name in names:
                for environment in range(q.shape[0]):
                    self._scene_collision.enable_obstacle(name, False, environment)
            self._disabled_obstacle_names = names
            self._disabled_num_envs = q.shape[0]

    def attach_from_scene(
        self,
        joint_states: JointState,
        obstacle_names: List[str],
        link_name: str = "attached_object",
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        if not obstacle_names:
            raise ValueError("obstacle_names must be non-empty")
        if self._scene_collision is None or self._scene_collision.scene_model is None:
            raise ValueError("attach_from_scene requires a configured scene_collision")
        scene = self._scene_collision.scene_model
        if isinstance(scene, list):
            scene = scene[0]
        obstacles = []
        for name in obstacle_names:
            value = scene.get_obstacle(name)
            if value is None:
                raise ValueError(f"obstacle '{name}' was not found")
            obstacles.append(value)
        self.attach(
            joint_states,
            obstacles,
            link_name,
            num_spheres,
            surface_radius,
            sphere_fit_type,
            world_objects_pose_offset,
            obstacle_names,
        )

    def detach(
        self,
        link_name: Optional[str] = None,
        enable_obstacle_names: Optional[List[str]] = None,
    ) -> None:
        link = link_name or self._attached_link_name
        if link is None:
            return
        self.kinematics_params.reset_link_spheres(link)
        names = enable_obstacle_names or self._disabled_obstacle_names
        if self._scene_collision is not None:
            for name in names:
                for environment in range(self._disabled_num_envs):
                    self._scene_collision.enable_obstacle(name, True, environment)
        if link == self._attached_link_name:
            self._attached_link_name = None
            self._disabled_obstacle_names = []
            self._disabled_num_envs = 0

    @staticmethod
    def _obstacles_to_trimesh(obstacles: List[Obstacle]):
        raise NotImplementedError(
            "trimesh conversion is optional; fit_spheres uses a dependency-free "
            "deterministic vertex representation"
        )

    @staticmethod
    def _get_num_envs(joint_states: JointState) -> int:
        return 1 if joint_states.position.ndim == 1 else joint_states.position.shape[0]
