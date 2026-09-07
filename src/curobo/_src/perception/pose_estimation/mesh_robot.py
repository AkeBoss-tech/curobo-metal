"""Portable rigid and articulated triangle meshes for pose estimation."""
# ruff: noqa: UP006, UP035, UP045

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    import trimesh
    import warp as wp

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose


def _require_warp():
    try:
        import warp as wp
    except ImportError as error:
        raise ImportError(
            "RobotMesh mesh/BVH construction requires the optional 'warp-lang' dependency"
        ) from error
    return wp


def _require_trimesh():
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "RobotMesh trimesh conversion requires the optional 'trimesh' dependency"
        ) from error
    return trimesh


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return requested


@dataclass
class SurfaceSampleCache:
    """Reusable triangle selections and barycentric coordinates."""

    face_indices: torch.Tensor
    bary_coords: torch.Tensor


class RobotMesh:
    """A triangle mesh whose tensor operations run on CPU or Metal.

    Warp cannot consume MPS tensors directly. A CPU Warp mesh is maintained
    as a compatibility BVH while vertices, FK, and sampling stay on the
    requested torch device. Updating an articulated mesh copies the new
    vertices into that stable backing mesh and refits it, preserving both the
    public mesh object and its ID.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        device: str = "cuda:0",
        kinematics: Optional[Kinematics] = None,
        link_vertex_ranges: Optional[List[Tuple[int, int]]] = None,
        link_names: Optional[List[str]] = None,
        link_vertices_local: Optional[List[torch.Tensor]] = None,
    ):
        self.device = _device(device)
        values = torch.as_tensor(vertices, dtype=torch.float32).to(self.device).contiguous()
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("vertices must be a floating [N, 3] tensor")
        indices = torch.as_tensor(faces, dtype=torch.int32).to(self.device).contiguous()
        if indices.ndim != 2 or indices.shape[-1] != 3:
            raise ValueError("faces must be an integer [F, 3] tensor")
        if len(indices) and bool(((indices < 0) | (indices >= len(values))).any().item()):
            raise ValueError("faces contain an out-of-range vertex index")

        self._vertices = values
        self._faces = indices
        self._kinematics = kinematics
        self._link_vertex_ranges = link_vertex_ranges
        self._link_names = link_names
        self._link_vertices_local = link_vertices_local
        self._is_articulated = kinematics is not None
        self._current_joints: Optional[torch.Tensor] = None
        self._sample_cache: Optional[SurfaceSampleCache] = None
        self._sample_generation = 0

        # Warp supports CPU on macOS, but not torch's MPS storage. Keep an
        # ordinary CPU tensor alive for zero-copy Warp ownership and update it
        # in-place before each BVH refit.
        wp = _require_warp()
        wp.init()
        self._vertices_cpu = self._vertices.detach().to("cpu").contiguous()
        self._faces_cpu = self._faces.detach().to("cpu").contiguous()
        self._vertices_wp = wp.from_torch(self._vertices_cpu, dtype=wp.vec3)
        self._mesh = wp.Mesh(
            points=self._vertices_wp,
            indices=wp.from_torch(self._faces_cpu.view(-1), dtype=wp.int32),
        )

    @staticmethod
    def _load_link_meshes(kinematics: Any) -> tuple[list[Any], list[str]]:
        """Load authored meshes, preferring the normal kinematics API."""
        try:
            meshes = kinematics.get_robot_link_meshes()
        except NotImplementedError:
            from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
            from curobo.content import get_assets_path

            params = kinematics.config.kinematics_config
            metadata = params.robot_cfg.metadata
            assets = Path(get_assets_path())
            parser = UrdfRobotParser(
                assets / metadata["urdf_path"],
                mesh_root=str(assets / metadata["asset_root_path"]),
            )
            names = list(params.mesh_link_names)
            meshes = [parser.get_link_mesh(name, use_collision_mesh=True) for name in names]
            pairs = [(mesh, name) for mesh, name in zip(meshes, names) if mesh is not None]
            if not pairs:
                raise ValueError("kinematics contains no loadable link meshes")
            return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

        names = list(kinematics.config.kinematics_config.mesh_link_names)
        pairs = [(mesh, name) for mesh, name in zip(meshes, names) if mesh is not None]
        if not pairs:
            raise ValueError("kinematics contains no loadable link meshes")
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    @classmethod
    def from_kinematics(
        cls,
        kinematics: Kinematics,
        device: str = "cuda:0",
        initial_joint_angles: Optional[torch.Tensor] = None,
    ) -> RobotMesh:
        if not hasattr(kinematics, "compute_kinematics") or not hasattr(kinematics, "dof"):
            raise TypeError("kinematics must expose compute_kinematics and dof")
        target = _device(device)
        link_meshes, link_names = cls._load_link_meshes(kinematics)
        all_vertices: list[torch.Tensor] = []
        all_faces: list[torch.Tensor] = []
        local_vertices: list[torch.Tensor] = []
        ranges: list[tuple[int, int]] = []
        offset = 0

        from curobo._src.types.device_cfg import DeviceCfg
        pose_device = DeviceCfg(device=target, dtype=torch.float32)
        for mesh in link_meshes:
            tm = mesh.get_trimesh_mesh(process=False)
            vertices = torch.as_tensor(tm.vertices, device=target, dtype=torch.float32)
            vertices = Pose.from_list(mesh.pose, pose_device).transform_points(vertices)
            faces = torch.as_tensor(tm.faces, device=target, dtype=torch.int32) + offset
            local_vertices.append(vertices)
            all_vertices.append(vertices)
            all_faces.append(faces)
            ranges.append((offset, offset + len(vertices)))
            offset += len(vertices)

        result = cls(
            torch.cat(all_vertices),
            torch.cat(all_faces),
            device=str(target),
            kinematics=kinematics,
            link_vertex_ranges=ranges,
            link_names=link_names,
            link_vertices_local=local_vertices,
        )
        q = (
            torch.zeros(kinematics.dof, device=target, dtype=torch.float32)
            if initial_joint_angles is None
            else torch.as_tensor(initial_joint_angles, device=target, dtype=torch.float32)
        )
        result.update(q)
        return result

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh, device: str = "cuda:0") -> RobotMesh:
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError("mesh must expose vertices and triangular faces")
        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.as_tensor(mesh.faces, dtype=torch.int32)
        return cls(vertices, faces, device)

    @property
    def vertices(self) -> torch.Tensor:
        return self._vertices

    @property
    def faces(self) -> torch.Tensor:
        return self._faces

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_faces(self) -> int:
        return len(self._faces)

    @property
    def current_joint_angles(self) -> Optional[torch.Tensor]:
        return self._current_joints

    @property
    def is_articulated(self) -> bool:
        return self._is_articulated

    @property
    def mesh(self) -> wp.Mesh:
        return self._mesh

    @property
    def mesh_id(self) -> wp.uint64:
        return self._mesh.id

    def get_dof(self) -> int:
        return 0 if self._kinematics is None else int(self._kinematics.dof)

    def update(self, joint_angles: torch.Tensor) -> None:
        if not self._is_articulated:
            return
        q = torch.as_tensor(joint_angles, device=self.device, dtype=torch.float32)
        if q.ndim == 2 and q.shape[0] == 1:
            q = q.squeeze(0)
        if q.ndim != 1 or q.shape[0] != self.get_dof():
            raise ValueError("joint_angles must have shape [dof] or [1, dof]")
        state = self._kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self._kinematics.joint_names)
        )
        if state.tool_poses is None:
            raise ValueError("KinematicsState.tool_poses is None")
        assert self._link_names is not None
        assert self._link_vertex_ranges is not None
        assert self._link_vertices_local is not None
        for name, vertex_range, local in zip(
            self._link_names, self._link_vertex_ranges, self._link_vertices_local
        ):
            world = state.tool_poses[name].transform_points(local)
            self._vertices[slice(*vertex_range)].copy_(world)

        self._vertices_cpu.copy_(self._vertices.detach().to("cpu"))
        self._mesh.refit()
        self._current_joints = q.clone()

    def _generate_sample_cache(self, n_points: int) -> SurfaceSampleCache:
        if n_points < 0:
            raise ValueError("n_points must be nonnegative")
        if self.n_faces == 0 and n_points:
            raise ValueError("cannot sample a mesh without faces")
        if n_points == 0:
            return SurfaceSampleCache(
                torch.empty(0, device=self.device, dtype=torch.long),
                torch.empty((0, 3), device=self.device, dtype=torch.float32),
            )
        faces = self._faces.to(torch.long)
        triangles = self._vertices[faces]
        areas = 0.5 * torch.linalg.vector_norm(
            torch.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=1,
            ),
            dim=1,
        )
        if not bool((areas.sum() > 0).item()):
            raise ValueError("cannot sample a zero-area mesh")
        # A low-discrepancy sequence gives repeatable, nested samples: the
        # first N points of an N+K request equal a direct N-point request.
        # This matters when a detector downsamples observations produced by
        # this same mesh and then asks the model for the smaller count.  A
        # fresh random draw creates artificial centimetre-scale alignment
        # error even at the identity pose.
        sequence = torch.arange(n_points, device=self.device, dtype=torch.float32) + 0.5
        phase = self._sample_generation * 0.3819660112501051
        face_u = torch.frac(sequence * 0.6180339887498949 + phase)
        cumulative = torch.cumsum(areas / areas.sum(), dim=0)
        cumulative[-1] = 1.0
        face_indices = torch.searchsorted(cumulative, face_u).clamp_max(self.n_faces - 1)
        r1 = torch.sqrt(torch.frac(sequence * 0.7548776662466927 + phase))
        r2 = torch.frac(sequence * 0.5698402909980532 + phase * 0.5)
        bary = torch.stack((1 - r1, r1 * (1 - r2), r1 * r2), dim=1)
        return SurfaceSampleCache(face_indices, bary)

    def sample_surface_points(
        self, n_points: int, resample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if resample:
            self._sample_generation += 1
        if (
            self._sample_cache is None
            or len(self._sample_cache.face_indices) != n_points
            or resample
        ):
            self._sample_cache = self._generate_sample_cache(n_points)
        face_indices = self._sample_cache.face_indices
        bary = self._sample_cache.bary_coords
        triangle = self._vertices[self._faces[face_indices].to(torch.long)]
        points = (triangle * bary.unsqueeze(-1)).sum(dim=1)
        normals = torch.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=1
        )
        normals = normals / torch.linalg.vector_norm(
            normals, dim=1, keepdim=True
        ).clamp_min(1e-8)
        return points, normals

    def get_trimesh(self) -> trimesh.Trimesh:
        trimesh = _require_trimesh()
        return trimesh.Trimesh(
            vertices=self._vertices.detach().cpu().numpy(),
            faces=self._faces.detach().cpu().numpy(),
            process=False,
        )


__all__ = ["RobotMesh", "SurfaceSampleCache"]
