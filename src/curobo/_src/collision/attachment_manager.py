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


class AttachmentManager:
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

    def fit_spheres(
        self,
        obstacles: List[Obstacle],
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        sphere_fit_type: SphereFitType = SphereFitType.MORPHIT,
    ) -> torch.Tensor:
        if not obstacles:
            raise ValueError("obstacles must be non-empty")
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
        return torch.cat((result.centers, result.radii[:, None]), dim=-1)

    def update(
        self,
        sphere_tensor: torch.Tensor,
        joint_states: JointState,
        link_name: str = "attached_object",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        values = torch.as_tensor(
            sphere_tensor,
            device=self._device_cfg.device,
            dtype=self._device_cfg.dtype,
        )
        if values.ndim != 2 or values.shape[-1] != 4:
            raise ValueError("sphere_tensor must have shape [num_spheres,4]")
        q = joint_states.position
        if q.ndim == 1:
            q = q.unsqueeze(0)
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
        if params.link_spheres.shape[0] != num_envs:
            params._link_spheres = params.link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()
            params.reference_link_spheres = params.reference_link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()
        if world_objects_pose_offset is not None:
            state = self._kinematics.compute_kinematics(
                JointState.from_position(q, joint_names=self._kinematics.joint_names)
            )
            link = (
                link_name if link_name in self._kinematics.tool_frames
                else self._kinematics.tool_frames[0]
            )
            link_pose = state.tool_poses.get_link_pose(link)
            object_to_link = link_pose.inverse().multiply(world_objects_pose_offset)
        else:
            object_to_link = None
        padding = values.new_zeros((len(indices), 4))
        padding[:, 3] = -100.0
        for environment in range(num_envs):
            current = values
            if object_to_link is not None:
                pose = Pose(
                    object_to_link.position[environment : environment + 1],
                    object_to_link.quaternion[environment : environment + 1],
                )
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
        values = self.fit_spheres(
            obstacles, num_spheres, surface_radius, sphere_fit_type
        )
        self.update(values, joint_states, link_name, world_objects_pose_offset)
        if disable_obstacle_names and self._scene_collision is not None:
            num_envs = self._get_num_envs(joint_states)
            for name in disable_obstacle_names:
                for environment in range(num_envs):
                    self._scene_collision.enable_obstacle(name, False, environment)
            self._disabled_obstacle_names = list(disable_obstacle_names)
            self._disabled_num_envs = num_envs

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
