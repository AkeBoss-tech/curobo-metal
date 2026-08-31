"""Portable geometry models used by pose-estimation workflows.

The CUDA implementation samples mesh surfaces once per robot link and applies
FK to that cache at query time. The same lifecycle is useful on CPU and MPS:
none of it requires a Warp mesh handle. This implementation deliberately
keeps the raw Warp-only mesh APIs out of scope while retaining the ordinary
tensor contract, including gradients through the supplied FK poses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

try:
    import trimesh as _raw_trimesh
except ImportError:
    _raw_trimesh = None

trimesh = _raw_trimesh
Kinematics = Any


def _mesh_value(mesh: Any) -> Any:
    """Resolve cuRobo/trimesh-like wrappers to an object with vertices/faces."""
    if hasattr(mesh, "get_trimesh_mesh"):
        mesh = mesh.get_trimesh_mesh()
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError("mesh must expose triangular vertices and faces")
    return mesh


def _mesh_tensors(mesh: Any, device_cfg: DeviceCfg) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = _mesh_value(mesh)
    vertices = torch.as_tensor(mesh.vertices, **device_cfg.as_torch_dict())
    faces = torch.as_tensor(mesh.faces, device=device_cfg.device, dtype=torch.long)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or not vertices.is_floating_point():
        raise ValueError("mesh vertices must be a floating [N, 3] tensor")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.numel() == 0:
        raise ValueError("mesh requires at least one triangular face")
    if bool(((faces < 0) | (faces >= len(vertices))).any().item()):
        raise ValueError("mesh faces contain an out-of-range vertex index")
    return vertices, faces


def _sample_surface(
    vertices: torch.Tensor, faces: torch.Tensor, n_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-weighted triangle samples with unit face normals."""
    if not isinstance(n_points, int) or n_points < 0:
        raise ValueError("n_points must be a non-negative integer")
    if n_points == 0:
        empty = vertices.new_empty((0, 3))
        return empty, empty.clone()
    triangles = vertices[faces]
    edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = torch.linalg.cross(edge_a, edge_b)
    doubled_area = torch.linalg.vector_norm(cross, dim=-1)
    if not bool((doubled_area > 0).any().item()):
        raise ValueError("mesh triangles must have non-zero area")
    selected = torch.multinomial(doubled_area, n_points, replacement=True)
    triangle = triangles.index_select(0, selected)
    normal = cross.index_select(0, selected)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(
        torch.finfo(vertices.dtype).eps
    )
    uv = torch.rand((n_points, 2), device=vertices.device, dtype=vertices.dtype)
    u, v = uv.unbind(-1)
    reflected = u + v > 1
    u, v = torch.where(reflected, 1 - u, u), torch.where(reflected, 1 - v, v)
    points = triangle[:, 0] + u[:, None] * (triangle[:, 1] - triangle[:, 0]) + v[:, None] * (
        triangle[:, 2] - triangle[:, 0]
    )
    return points, normal


def _mesh_volume(mesh: Any, vertices: torch.Tensor, faces: torch.Tensor) -> float:
    """Use a mesh volume when supplied, otherwise a stable area proxy."""
    value = getattr(_mesh_value(mesh), "volume", None)
    if value is not None:
        try:
            value = abs(float(value))
        except (TypeError, ValueError):
            value = 0.0
    if not value or value <= 0:
        triangle = vertices[faces]
        cross = torch.linalg.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        value = float((torch.linalg.vector_norm(cross, dim=-1).sum() / 2).item())
    return value


def _mesh_pose(mesh: Any, device_cfg: DeviceCfg) -> Pose | None:
    value = getattr(mesh, "pose", None)
    if value is None:
        return None
    if isinstance(value, Pose):
        return value.to(device_cfg=device_cfg)
    return Pose.from_list(list(value), device_cfg=device_cfg)


def _link_names(robot_model: Any, count: int) -> list[str]:
    config = getattr(robot_model, "config", None)
    kinematics = getattr(config, "kinematics_config", None)
    names = getattr(kinematics, "mesh_link_names", None)
    if names is None:
        names = getattr(robot_model, "mesh_link_names", None)
    if names is None:
        names = [f"link_{index}" for index in range(count)]
    names = list(names)
    if len(names) != count or len(set(names)) != len(names):
        raise ValueError("robot mesh link names must be unique and match its mesh count")
    return names


def _state_pose(state: Any, link_name: str) -> Pose:
    poses = getattr(state, "tool_poses", None)
    if poses is None:
        raise ValueError("kinematics state has no tool poses")
    if isinstance(poses, dict):
        pose = poses.get(link_name)
    elif hasattr(poses, "get_link_pose"):
        pose = poses.get_link_pose(link_name)
    else:
        try:
            pose = poses[link_name]
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f"kinematics state has no pose for link {link_name!r}") from error
    if not isinstance(pose, Pose):
        raise TypeError("kinematics link poses must be Pose instances")
    if pose.position is None or pose.quaternion is None:
        raise ValueError("kinematics link pose must be populated")
    if pose.position.reshape(-1, 3).shape[0] != 1:
        raise ValueError("articulated surface sampling supports one joint configuration at a time")
    return pose


class RigidObjectGeometry:
    """A zero-DOF triangle mesh with CPU/MPS surface sampling."""

    def __init__(self, mesh: trimesh.Trimesh, device_cfg: DeviceCfg = DeviceCfg()):
        self.mesh = mesh
        self._tensor_args = device_cfg

    def update(self, joint_angles: torch.Tensor):
        del joint_angles

    def sample_surface_points(self, n_points: int):
        vertices, faces = _mesh_tensors(self.mesh, self._tensor_args)
        return _sample_surface(vertices, faces, n_points)

    def get_dof(self) -> int:
        return 0

    @property
    def device_cfg(self):
        return self._tensor_args


class ArticulatedRobotGeometry:
    """Cached link-frame surface samples transformed by regular PyTorch FK."""

    def __init__(
        self,
        robot_model: Kinematics,
        device_cfg: DeviceCfg = DeviceCfg(),
        points_per_cubic_meter: float = 150000.0,
        min_points_per_link: int = 50,
        max_points_per_link: int = 200,
    ):
        if (
            points_per_cubic_meter < 0
            or min_points_per_link < 1
            or max_points_per_link < min_points_per_link
        ):
            raise ValueError("invalid articulated surface-sampling bounds")
        if not hasattr(robot_model, "get_robot_link_meshes") or not hasattr(
            robot_model, "compute_kinematics"
        ):
            raise NotImplementedError("robot model must expose link meshes and compute_kinematics")
        self.robot_model = robot_model
        self._tensor_args = device_cfg
        names = getattr(robot_model, "joint_names", None)
        self._n_dof = len(names) if names is not None else int(getattr(robot_model, "dof", 0))
        if self._n_dof < 0:
            raise ValueError("robot model dof must be non-negative")
        self._initialize_cached_points(
            points_per_cubic_meter, min_points_per_link, max_points_per_link
        )
        self._current_config: torch.Tensor | None = None

    def _initialize_cached_points(
        self, points_per_cubic_meter: float, min_points_per_link: int, max_points_per_link: int
    ) -> None:
        meshes = list(self.robot_model.get_robot_link_meshes())
        if not meshes:
            raise ValueError("robot model has no link meshes to sample")
        self.cached_link_names = _link_names(self.robot_model, len(meshes))
        self.cached_link_points: list[torch.Tensor] = []
        self.cached_link_normals: list[torch.Tensor] = []
        for mesh in meshes:
            vertices, faces = _mesh_tensors(mesh, self._tensor_args)
            count = min(
                max_points_per_link,
                max(
                    min_points_per_link,
                    int(_mesh_volume(mesh, vertices, faces) * points_per_cubic_meter),
                ),
            )
            points, normals = _sample_surface(vertices, faces, count)
            local_pose = _mesh_pose(mesh, self._tensor_args)
            if local_pose is not None:
                points = local_pose.transform_points(points)
                rotation = local_pose.get_rotation().reshape(-1, 3, 3)
                if rotation.shape[0] != 1:
                    raise ValueError("link mesh local pose must contain one transform")
                normals = normals @ rotation[0].transpose(-1, -2)
            self.cached_link_points.append(points)
            self.cached_link_normals.append(normals)

    def update(self, joint_angles: torch.Tensor):
        if not isinstance(joint_angles, torch.Tensor) or joint_angles.ndim not in (1, 2):
            raise ValueError("joint_angles must be a [dof] or [1, dof] tensor")
        config = joint_angles.unsqueeze(0) if joint_angles.ndim == 1 else joint_angles
        if config.shape != (1, self._n_dof):
            raise ValueError("articulated surface sampling supports one [dof] configuration")
        self._current_config = config.to(**self._tensor_args.as_torch_dict())

    def sample_surface_points(self, n_points: int):
        if not isinstance(n_points, int) or n_points < 0:
            raise ValueError("n_points must be a non-negative integer")
        if self._current_config is None:
            raise ValueError("Must call update() with joint angles before sampling")
        joint_names = getattr(self.robot_model, "joint_names", None)
        state = self.robot_model.compute_kinematics(
            JointState.from_position(self._current_config, joint_names=joint_names)
        )
        points: list[torch.Tensor] = []
        normals: list[torch.Tensor] = []
        for name, link_points, link_normals in zip(
            self.cached_link_names, self.cached_link_points, self.cached_link_normals
        ):
            pose = _state_pose(state, name).to(device_cfg=self._tensor_args)
            points.append(pose.transform_points(link_points))
            rotation = pose.get_rotation().reshape(-1, 3, 3)[0]
            normals.append(link_normals @ rotation.transpose(-1, -2))
        result_points, result_normals = torch.cat(points), torch.cat(normals)
        if len(result_points) > n_points:
            indices = torch.randperm(len(result_points), device=result_points.device)[:n_points]
            result_points, result_normals = result_points[indices], result_normals[indices]
        return result_points, result_normals

    def get_dof(self) -> int:
        return self._n_dof

    @property
    def device_cfg(self):
        return self._tensor_args
