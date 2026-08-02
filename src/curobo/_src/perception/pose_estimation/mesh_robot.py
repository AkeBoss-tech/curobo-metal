"""Portable triangle-mesh representation for pose-estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def _device(value: str) -> str:
    if str(value).startswith("cuda"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return str(value)


@dataclass
class SurfaceSampleCache:
    points: torch.Tensor
    normals: torch.Tensor


class RobotMesh:
    """A CPU/MPS mesh with deterministic area-free surface sampling.

    Warp mesh handles and IDs deliberately remain unavailable.  This class is
    useful without them for centroid registration, point-to-plane detection,
    and mesh-based debug rendering.
    """

    def __init__(
        self, vertices: torch.Tensor, faces: torch.Tensor, device: str = "cuda:0",
        kinematics=None, link_vertex_ranges: Optional[List[Tuple[int, int]]] = None,
        link_names: Optional[List[str]] = None,
        link_vertices_local: Optional[List[torch.Tensor]] = None,
    ) -> None:
        target = _device(device)
        values = torch.as_tensor(vertices, device=target)
        if values.ndim != 2 or values.shape[-1] != 3 or not values.is_floating_point():
            raise ValueError("vertices must be a floating [N, 3] tensor")
        indices = torch.as_tensor(faces, device=target, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[-1] != 3:
            raise ValueError("faces must be an integer [F, 3] tensor")
        if len(indices) and bool(((indices < 0) | (indices >= len(values))).any().item()):
            raise ValueError("faces contain an out-of-range vertex index")
        self._vertices = values
        self._faces = indices
        self.kinematics = kinematics
        self.link_vertex_ranges = link_vertex_ranges
        self.link_names = link_names
        self.link_vertices_local = link_vertices_local
        self._current_joint_angles: Optional[torch.Tensor] = None
        self._sample_cache: dict[int, SurfaceSampleCache] = {}

    @classmethod
    def from_kinematics(cls, kinematics, device: str = "cuda:0", initial_joint_angles: Optional[torch.Tensor] = None) -> "RobotMesh":
        if not hasattr(kinematics, "get_robot_as_spheres"):
            raise TypeError("kinematics must expose get_robot_as_spheres")
        dof = getattr(kinematics, "dof", None)
        if dof is None:
            raise TypeError("kinematics must expose dof")
        q = torch.zeros((1, dof), device=_device(device), dtype=torch.float32) if initial_joint_angles is None else initial_joint_angles.to(_device(device))
        spheres = kinematics.get_robot_as_spheres(q)
        values = torch.tensor([[*sphere.pose[:3]] for sphere in spheres[0]], device=_device(device), dtype=q.dtype)
        # A sphere-only kinematics model does not contain mesh faces.  A small
        # empty face tensor accurately preserves that boundary.
        result = cls(values, torch.empty((0, 3), device=values.device, dtype=torch.long), device, kinematics=kinematics)
        result._current_joint_angles = q.clone()
        return result

    @classmethod
    def from_trimesh(cls, mesh, device: str = "cuda:0") -> "RobotMesh":
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError("mesh must expose vertices and triangular faces")
        return cls(torch.as_tensor(mesh.vertices), torch.as_tensor(mesh.faces), device)

    @property
    def vertices(self) -> torch.Tensor:
        return self._vertices

    @property
    def faces(self) -> torch.Tensor:
        return self._faces

    @property
    def n_vertices(self) -> int:
        return int(self._vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self._faces.shape[0])

    @property
    def current_joint_angles(self) -> Optional[torch.Tensor]:
        return self._current_joint_angles

    @property
    def is_articulated(self) -> bool:
        return self.kinematics is not None

    @property
    def mesh(self):
        raise NotImplementedError("raw Warp mesh handles are unavailable; use vertices and faces")

    @property
    def mesh_id(self):
        raise NotImplementedError("raw Warp mesh IDs are unavailable; use vertices and faces")

    def get_dof(self) -> int:
        return int(getattr(self.kinematics, "dof", 0))

    def update(self, joint_angles: torch.Tensor) -> None:
        if not self.is_articulated:
            if joint_angles.numel() != 0:
                raise ValueError("a rigid RobotMesh has no joint angles")
            return
        if joint_angles.shape[-1] != self.get_dof():
            raise ValueError("joint angle width does not match kinematics dof")
        self._current_joint_angles = joint_angles.clone()

    def sample_surface_points(self, n_points: int, resample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_points < 0:
            raise ValueError("n_points must be nonnegative")
        if n_points in self._sample_cache and not resample:
            cached = self._sample_cache[n_points]
            return cached.points, cached.normals
        if n_points == 0:
            empty = self._vertices.new_empty((0, 3))
            return empty, empty.clone()
        if self.n_faces == 0:
            if self.n_vertices == 0:
                raise ValueError("cannot sample an empty mesh")
            points = self._vertices[torch.arange(n_points, device=self._vertices.device) % self.n_vertices]
            normals = torch.zeros_like(points)
        else:
            faces = self._faces[torch.arange(n_points, device=self._faces.device) % self.n_faces]
            triangle = self._vertices[faces]
            points = triangle.mean(dim=1)
            normals = torch.linalg.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
            normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(torch.finfo(points.dtype).eps)
        self._sample_cache[n_points] = SurfaceSampleCache(points, normals)
        return points, normals

    def get_trimesh(self):
        try:
            import trimesh
        except ModuleNotFoundError as error:
            raise NotImplementedError("get_trimesh requires optional trimesh") from error
        return trimesh.Trimesh(vertices=self._vertices.detach().cpu().numpy(), faces=self._faces.detach().cpu().numpy(), process=False)


__all__ = ["RobotMesh", "SurfaceSampleCache"]
