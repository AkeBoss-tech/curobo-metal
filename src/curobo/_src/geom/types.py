"""Pinned cuRobo geometry records backed by portable collision data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _portable_tensor(value: Any, device_cfg: DeviceCfg) -> torch.Tensor:
    """Convert a geometry value while preserving an existing tensor's graph."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device_cfg.device, dtype=device_cfg.dtype)
    return torch.as_tensor(value, device=device_cfg.device, dtype=device_cfg.dtype)


def _offset_pose(base_pose: Sequence[float] | None, local_translation: torch.Tensor, cfg: DeviceCfg) -> list[float]:
    """Compose a local translation with an obstacle pose in portable torch."""
    base = Pose.from_list(list(base_pose or [0, 0, 0, 1, 0, 0, 0]), cfg)
    offset = Pose(local_translation.to(**cfg.as_torch_dict()), torch.tensor([1, 0, 0, 0], **cfg.as_torch_dict()))
    return base.multiply(offset).tolist()


def _copy_visual_fields(source: "Obstacle", target: "Obstacle") -> "Obstacle":
    """Copy portable visual metadata when converting an obstacle to a mesh.

    Geometry conversion is used by both collision-scene construction and
    mapper export.  Keeping this in one place avoids silently dropping
    per-vertex fields just because the caller is running without trimesh.
    """
    for name in (
        "texture_id", "texture", "vertex_colors", "vertex_normals",
        "texture_uvs", "texture_image", "face_colors",
    ):
        if hasattr(source, name):
            value = getattr(source, name)
            if isinstance(value, torch.Tensor):
                value = value.clone()
            elif isinstance(value, list):
                value = value.copy()
            setattr(target, name, value)
    return target


def _clone_value(value: Any) -> Any:
    """Clone mutable/tensor scene values without detaching autograd graphs."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _triangulate_polygon_faces(
    faces: Sequence[Any] | torch.Tensor,
    face_counts: Sequence[int] | torch.Tensor | None = None,
) -> list[list[int]]:
    """Fan-triangulate flat or nested polygon buffers deterministically."""
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().tolist()
    if face_counts is not None:
        if isinstance(face_counts, torch.Tensor):
            face_counts = face_counts.detach().cpu().tolist()
        flat = [int(index) for index in faces]
        triangles: list[list[int]] = []
        offset = 0
        for count in face_counts:
            count = int(count)
            if count < 3 or offset + count > len(flat):
                raise ValueError("face_counts must partition a flat buffer into polygons of at least three vertices")
            polygon = flat[offset : offset + count]
            triangles.extend([[polygon[0], polygon[index], polygon[index + 1]] for index in range(1, count - 1)])
            offset += count
        if offset != len(flat):
            raise ValueError("face_counts must consume every entry in faces")
        return triangles
    triangles = []
    for face in faces:
        polygon = [int(index) for index in face]
        if len(polygon) < 3:
            raise ValueError("polygon faces must contain at least three indices")
        triangles.extend([[polygon[0], polygon[index], polygon[index + 1]] for index in range(1, len(polygon) - 1)])
    return triangles


def _primitive_mesh(vertices: torch.Tensor, faces: torch.Tensor, obstacle: "Obstacle") -> "Mesh":
    """Construct a local-space mesh retaining portable obstacle metadata."""
    result = Mesh(
        obstacle.name,
        pose=list(obstacle._pose_or_identity()),
        vertices=vertices.detach().cpu().tolist(),
        faces=faces.detach().cpu().tolist(),
        color=None if obstacle.color is None else list(obstacle.color),
        texture_id=obstacle.texture_id,
        texture=obstacle.texture,
        material=obstacle.material,
        device_cfg=obstacle.device_cfg,
    )
    return _copy_visual_fields(obstacle, result)  # type: ignore[return-value]


def _sphere_surface(radius: float, *, latitude_segments: int = 8, longitude_segments: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic latitude/longitude sphere mesh in local space."""
    if latitude_segments < 2 or longitude_segments < 3:
        raise ValueError("sphere tessellation requires at least 2 latitude and 3 longitude segments")
    dtype = torch.float64
    theta = torch.arange(longitude_segments, dtype=dtype) * (2.0 * torch.pi / longitude_segments)
    rings = torch.arange(1, latitude_segments, dtype=dtype) * (torch.pi / latitude_segments)
    sin_phi = torch.sin(rings).unsqueeze(-1)
    ring_vertices = torch.stack(
        (
            sin_phi * torch.cos(theta),
            sin_phi * torch.sin(theta),
            torch.cos(rings).unsqueeze(-1).expand(-1, longitude_segments),
        ),
        dim=-1,
    ).reshape(-1, 3)
    vertices = torch.cat(
        (
            torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
            ring_vertices,
            torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype),
        ),
        dim=0,
    ) * float(radius)
    faces: list[list[int]] = []
    for index in range(longitude_segments):
        following = (index + 1) % longitude_segments
        faces.append([0, 1 + following, 1 + index])
    for ring in range(latitude_segments - 2):
        start = 1 + ring * longitude_segments
        next_start = start + longitude_segments
        for index in range(longitude_segments):
            following = (index + 1) % longitude_segments
            faces.extend([[start + index, start + following, next_start + following], [start + index, next_start + following, next_start + index]])
    bottom = vertices.shape[0] - 1
    last = 1 + (latitude_segments - 2) * longitude_segments
    for index in range(longitude_segments):
        following = (index + 1) % longitude_segments
        faces.append([bottom, last + index, last + following])
    return vertices, torch.tensor(faces, dtype=torch.long)


def _cylinder_surface(radius: float, height: float, *, segments: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a closed z-axis cylinder mesh in local space."""
    if segments < 3:
        raise ValueError("cylinder tessellation requires at least three segments")
    theta = torch.arange(segments, dtype=torch.float64) * (2.0 * torch.pi / segments)
    circle = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
    lower = torch.cat((circle, torch.full((segments, 1), -0.5 * height, dtype=torch.float64)), dim=-1)
    upper = torch.cat((circle, torch.full((segments, 1), 0.5 * height, dtype=torch.float64)), dim=-1)
    vertices = torch.cat((lower, upper, torch.tensor([[0.0, 0.0, -0.5 * height], [0.0, 0.0, 0.5 * height]], dtype=torch.float64)), dim=0)
    lower_center, upper_center = 2 * segments, 2 * segments + 1
    faces: list[list[int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend([[index, following, segments + following], [index, segments + following, segments + index], [lower_center, following, index], [upper_center, segments + index, segments + following]])
    return vertices, torch.tensor(faces, dtype=torch.long)


def tensor_sphere(
    pt: Sequence[float] | torch.Tensor,
    radius: float | torch.Tensor,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Return portable ``[x, y, z, radius]`` sphere storage."""
    point = _portable_tensor(pt, device_cfg)
    if point.shape[-1:] != (3,):
        raise ValueError("sphere point must end in dimension 3")
    radius_t = _portable_tensor(radius, device_cfg).reshape(1)
    if tensor is None:
        return torch.cat((point, radius_t), dim=-1)
    if tensor.shape[-1:] != (4,):
        raise ValueError("sphere tensor must end in dimension 4")
    tensor[..., :3].copy_(point)
    tensor[..., 3].copy_(radius_t.expand_as(tensor[..., 3]))
    return tensor


def tensor_capsule(
    base: Sequence[float] | torch.Tensor,
    tip: Sequence[float] | torch.Tensor,
    radius: float | torch.Tensor,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Return portable ``[base_xyz, tip_xyz, radius]`` capsule storage."""
    base_t, tip_t = _portable_tensor(base, device_cfg), _portable_tensor(tip, device_cfg)
    if base_t.shape[-1:] != (3,) or tip_t.shape[-1:] != (3,):
        raise ValueError("capsule base and tip must end in dimension 3")
    radius_t = _portable_tensor(radius, device_cfg).reshape(1)
    if tensor is None:
        return torch.cat((base_t, tip_t, radius_t), dim=-1)
    if tensor.shape[-1:] != (7,):
        raise ValueError("capsule tensor must end in dimension 7")
    tensor[..., :3].copy_(base_t)
    tensor[..., 3:6].copy_(tip_t)
    tensor[..., 6].copy_(radius_t.expand_as(tensor[..., 6]))
    return tensor


def tensor_cube(
    pose: Sequence[float] | torch.Tensor,
    dims: Sequence[float] | torch.Tensor,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> list[torch.Tensor]:
    """Return ``[dimensions, inverse_pose]`` for one centered cuboid."""
    pose_t = _portable_tensor(pose, device_cfg)
    dims_t = _portable_tensor(dims, device_cfg)
    if pose_t.shape != (7,) or dims_t.shape != (3,):
        raise ValueError("cube pose and dims must have shapes [7] and [3]")
    forward = Pose(pose_t[:3], pose_t[3:])
    return [dims_t, forward.inverse().get_pose_vector().squeeze(0)]


def batch_tensor_cube(
    pose: Sequence[Sequence[float]] | torch.Tensor,
    dims: Sequence[Sequence[float]] | torch.Tensor,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> list[torch.Tensor]:
    """Vectorized ``tensor_cube`` for ``[batch, 7]`` cuboid poses."""
    pose_t = _portable_tensor(pose, device_cfg)
    dims_t = _portable_tensor(dims, device_cfg)
    if pose_t.ndim != 2 or pose_t.shape[-1] != 7 or dims_t.shape != (pose_t.shape[0], 3):
        raise ValueError("cube batches require pose [B,7] and dims [B,3]")
    forward = Pose(pose_t[:, :3], pose_t[:, 3:])
    return [dims_t, forward.inverse().get_pose_vector()]


@dataclass
class Material:
    metallic: float = 0.0
    roughness: float = 0.4


@dataclass
class Obstacle:
    name: str
    pose: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    color: Optional[List[float]] = None
    texture_id: Optional[str] = None
    texture: Optional[str] = None
    material: Material = field(default_factory=Material)
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    def __post_init__(self) -> None:
        if self.pose is not None and len(self.pose) != 7:
            raise ValueError("pose must be [x, y, z, qw, qx, qy, qz]")

    def _pose_or_identity(self) -> list[float]:
        return self.pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def clone(self) -> "Obstacle":
        """Return an independent portable value copy.

        Scene mutation/caching routinely clones worlds before applying an
        attachment or voxel update.  Tensor fields remain differentiable: a
        clone participates in the same graph rather than silently detaching.
        """
        values = {
            name: _clone_value(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        return type(self)(**values)

    def get_transform_matrix(self) -> np.ndarray:
        """Return a serialisable homogeneous transform without trimesh."""
        matrix = Pose.from_list(self._pose_or_identity(), self.device_cfg).get_matrix()
        return matrix.squeeze(0).detach().cpu().numpy()

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True, transform_with_pose: bool = False):
        raise NotImplementedError(
            "trimesh/asset conversion is optional and is not part of the portable Metal runtime; "
            "use get_mesh() or explicit vertices/faces"
        )

    def save_as_mesh(self, file_path: str, transform_with_pose: bool = False) -> None:
        """Write an explicit OBJ mesh without a trimesh dependency.

        This only supports meshes representable by the portable primitive
        tessellators; USD/texture/scene graph export remains external.
        """
        mesh = self.get_mesh()
        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float64)
        if transform_with_pose:
            pose = Pose.from_list(mesh._pose_or_identity(), self.device_cfg)
            vertices = pose.transform_points(vertices.to(**self.device_cfg.as_torch_dict())).detach().cpu()
        faces = torch.as_tensor(mesh.faces, dtype=torch.long)
        with Path(file_path).open("w", encoding="utf-8") as output:
            for vertex in vertices.tolist():
                output.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            for face in faces.tolist():
                output.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")

    def get_mesh(self, process: bool = True) -> "Mesh":
        raise NotImplementedError(f"{type(self).__name__} cannot be converted to a portable triangle mesh")

    def get_cuboid(self) -> "Cuboid":
        raise NotImplementedError(f"{type(self).__name__} has no portable cuboid approximation")

    def get_sphere(self, n: int = 1) -> "Sphere":
        if n != 1:
            raise NotImplementedError("portable obstacle bounding-sphere conversion currently supports one sphere")
        cuboid = self.get_cuboid()
        return Sphere(name=f"{self.name}_sphere", pose=cuboid.pose, radius=0.5 * min(cuboid.dims), device_cfg=self.device_cfg)

    def get_bounding_spheres(self, num_spheres: int | None = None, surface_radius: float = 0.002,
                             fit_type=None, pre_transform_pose: Pose | None = None,
                             device_cfg: DeviceCfg = DeviceCfg()) -> list["Sphere"]:
        if num_spheres not in (None, 1):
            raise NotImplementedError("multi-sphere fitting requires the optional fitting backend")
        if surface_radius < 0:
            raise ValueError("surface_radius must be nonnegative")
        sphere = self.get_sphere()
        sphere.radius += surface_radius
        if pre_transform_pose is not None:
            center = pre_transform_pose.transform_points(
                torch.as_tensor(sphere.pose[:3], **device_cfg.as_torch_dict())
            ).reshape(-1, 3)[0]
            quaternion = pre_transform_pose.multiply(Pose.from_list(sphere.pose, device_cfg)).quaternion.reshape(-1, 4)[0]
            sphere.pose = torch.cat((center, quaternion)).detach().cpu().tolist()
        return [sphere]


@dataclass
class Cuboid(Obstacle):
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pose is None:
            raise ValueError("Cuboid Obstacle requires Pose")
        if len(self.dims) != 3 or any(x < 0 for x in self.dims):
            raise ValueError("dims must contain three nonnegative lengths")

    def get_cuboid(self) -> "Cuboid":
        return Cuboid(self.name, list(self.pose), list(self.dims), color=self.color, material=self.material, device_cfg=self.device_cfg)

    def get_mesh(self, process: bool = True) -> "Mesh":
        half = torch.as_tensor(self.dims, dtype=torch.float32) * 0.5
        signs = torch.tensor([
            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
            [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1],
        ], dtype=torch.float32)
        faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                 [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        return _primitive_mesh(signs * half, torch.tensor(faces, dtype=torch.long), self)


@dataclass
class Sphere(Obstacle):
    radius: float = 0.0
    position: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.position is not None:
            if len(self.position) != 3:
                raise ValueError("position must contain three values")
            self.pose = list(self.position) + [1, 0, 0, 0]
        super().__post_init__()
        if self.radius < 0:
            raise ValueError("radius must be nonnegative")
        if self.pose is not None:
            self.position = list(self.pose[:3])

    def get_cuboid(self) -> Cuboid:
        return Cuboid(self.name, self._pose_or_identity(), dims=[2*self.radius]*3, color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def get_mesh(self, process: bool = True) -> "Mesh":
        vertices, faces = _sphere_surface(self.radius)
        return _primitive_mesh(vertices, faces, self)


@dataclass
class Capsule(Obstacle):
    radius: float = 0.0
    base: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tip: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.radius < 0 or len(self.base) != 3 or len(self.tip) != 3:
            raise ValueError("Capsule requires a nonnegative radius and 3D base/tip")

    def get_cuboid(self) -> Cuboid:
        base, tip = torch.tensor(self.base), torch.tensor(self.tip)
        lower, upper = torch.minimum(base, tip) - self.radius, torch.maximum(base, tip) + self.radius
        center = (lower + upper) * 0.5
        return Cuboid(self.name, _offset_pose(self.pose, center, self.device_cfg), (upper-lower).tolist(), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def get_mesh(self, process: bool = True) -> "Mesh":
        base = torch.as_tensor(self.base, dtype=torch.float64)
        tip = torch.as_tensor(self.tip, dtype=torch.float64)
        axis = tip - base
        length = torch.linalg.vector_norm(axis)
        # A zero-length capsule is exactly a sphere.  This is common for a
        # sphere encoded in a capsule buffer and should not create NaNs.
        if bool(length <= torch.finfo(length.dtype).eps):
            vertices, faces = _sphere_surface(self.radius)
            vertices = vertices + base
            return _primitive_mesh(vertices, faces, self)
        direction = axis / length
        reference = torch.tensor([1.0, 0.0, 0.0], dtype=base.dtype)
        if bool(torch.abs(direction[0]) > 0.9):
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=base.dtype)
        first = torch.linalg.cross(direction, reference)
        first = first / torch.linalg.vector_norm(first)
        second = torch.linalg.cross(direction, first)
        segments, hemispheres = 16, 6
        angle = torch.arange(segments, dtype=base.dtype) * (2.0 * torch.pi / segments)
        radial = torch.cos(angle).unsqueeze(-1) * first + torch.sin(angle).unsqueeze(-1) * second
        rings: list[torch.Tensor] = [base - direction * self.radius]
        # Bottom hemisphere excludes its pole and its equator; the two
        # cylinder rings below provide the equators exactly once.
        for index in range(1, hemispheres):
            phi = torch.as_tensor(-0.5 * torch.pi + index * (0.5 * torch.pi / hemispheres), dtype=base.dtype)
            rings.append(base + direction * (self.radius * torch.sin(phi)) + radial * (self.radius * torch.cos(phi)))
        rings.append(base + radial * self.radius)
        rings.append(tip + radial * self.radius)
        for index in range(1, hemispheres):
            phi = torch.as_tensor(index * (0.5 * torch.pi / hemispheres), dtype=base.dtype)
            rings.append(tip + direction * (self.radius * torch.sin(phi)) + radial * (self.radius * torch.cos(phi)))
        rings.append(tip + direction * self.radius)
        vertices: list[torch.Tensor] = []
        ring_starts: list[int] = []
        for ring in rings:
            ring_starts.append(len(vertices))
            if ring.ndim == 1:
                vertices.append(ring)
            else:
                vertices.extend(ring.unbind(0))
        faces: list[list[int]] = []
        for ring_index in range(len(rings) - 1):
            lower, upper = rings[ring_index], rings[ring_index + 1]
            lower_start, upper_start = ring_starts[ring_index], ring_starts[ring_index + 1]
            if lower.ndim == 1:
                for index in range(segments):
                    faces.append([lower_start, upper_start + (index + 1) % segments, upper_start + index])
            elif upper.ndim == 1:
                for index in range(segments):
                    faces.append([upper_start, lower_start + index, lower_start + (index + 1) % segments])
            else:
                for index in range(segments):
                    following = (index + 1) % segments
                    faces.extend([[lower_start + index, lower_start + following, upper_start + following], [lower_start + index, upper_start + following, upper_start + index]])
        return _primitive_mesh(torch.stack(vertices), torch.tensor(faces, dtype=torch.long), self)


@dataclass
class Cylinder(Obstacle):
    radius: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.radius < 0 or self.height < 0:
            raise ValueError("Cylinder radius and height must be nonnegative")

    def get_cuboid(self) -> Cuboid:
        return Cuboid(self.name, self._pose_or_identity(), [2*self.radius, 2*self.radius, self.height],
                      color=self.color, material=self.material, device_cfg=self.device_cfg)

    def get_mesh(self, process: bool = True) -> "Mesh":
        vertices, faces = _cylinder_surface(self.radius, self.height)
        return _primitive_mesh(vertices, faces, self)


@dataclass
class PointCloud(Obstacle):
    points: Any = None
    points_features: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.points is None:
            raise ValueError("PointCloud requires points")
        points = torch.as_tensor(self.points)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("points must end in dimension 3")
        if self.scale is not None:
            dtype = points.dtype if points.is_floating_point() else torch.get_default_dtype()
            scale = torch.as_tensor(self.scale, dtype=dtype, device=points.device)
            if scale.shape not in (torch.Size([]), torch.Size([3])):
                raise ValueError("point-cloud scale must be scalar or contain three values")
            self.points = points.to(dtype=dtype) * scale
            self.scale = None

    def get_mesh_data(self, process: bool = True):
        mesh = Mesh.from_pointcloud(torch.as_tensor(self.points).reshape(-1, 3), name=self.name, pose=self._pose_or_identity())
        return mesh.get_mesh_data(process)

    def get_mesh(self, process: bool = True) -> "Mesh":
        mesh = Mesh.from_pointcloud(
            torch.as_tensor(self.points).reshape(-1, 3),
            name=self.name,
            pose=self._pose_or_identity(),
        )
        mesh.color = None if self.color is None else list(self.color)
        mesh.material = self.material
        mesh.device_cfg = self.device_cfg
        return mesh

    @staticmethod
    def from_camera_observation(camera_obs, name: str = "pc_obstacle", pose: list[float] | None = None) -> "PointCloud":
        if not hasattr(camera_obs, "get_pointcloud"):
            raise TypeError("camera observation must provide get_pointcloud()")
        return PointCloud(name=name, pose=pose, points=camera_obs.get_pointcloud())


@dataclass
class Mesh(Obstacle):
    file_path: Optional[str] = None
    file_string: Optional[str] = None
    urdf_path: Optional[str] = None
    vertices: Optional[Any] = None
    faces: Optional[Any] = None
    vertex_colors: Optional[Any] = None
    vertex_normals: Optional[Any] = None
    texture_uvs: Optional[Any] = None
    texture_image: Optional[Any] = None
    face_colors: Optional[Any] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.file_path is not None and self.vertices is None:
            raise NotImplementedError("file-backed meshes are not supported by the portable backend")
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh requires vertices and faces")
        vertices, faces = torch.as_tensor(self.vertices), torch.as_tensor(self.faces)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [N, 3]")
        if faces.ndim == 1 and faces.numel() == 3:
            faces = faces.reshape(1, 3)
            self.faces = faces
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError("portable world collision supports triangulated faces [F, 3]")
        if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
            raise ValueError("mesh face indices are outside vertices")
        if self.scale is not None:
            dtype = vertices.dtype if vertices.is_floating_point() else torch.get_default_dtype()
            scale = torch.as_tensor(self.scale, dtype=dtype, device=vertices.device)
            if scale.shape not in (torch.Size([]), torch.Size([3])):
                raise ValueError("mesh scale must be scalar or contain three values")
            # Preserve an incoming tensor's graph and device.  The dataclass
            # remains serialisable because list/ndarray callers retain their
            # original representation through the multiplication below.
            self.vertices = vertices.to(dtype=dtype) * scale
            self.scale = None

    @classmethod
    def from_polygon_faces(
        cls,
        name: str,
        vertices,
        faces,
        face_counts: Sequence[int] | torch.Tensor | None = None,
        pose: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        color: Optional[List[float]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs,
    ) -> "Mesh":
        """Create triangulated mesh storage from flat or nested polygons.

        The pinned V2 entry point supplies a flat face buffer plus
        ``face_counts``.  Earlier portable callers supplied a nested list.
        Supporting both is deterministic and avoids an incidental API break.
        """
        return cls(
            name=name,
            vertices=vertices,
            faces=_triangulate_polygon_faces(faces, face_counts),
            pose=pose,
            scale=scale,
            color=color,
            device_cfg=device_cfg,
            **kwargs,
        )

    @staticmethod
    def from_pointcloud(
        pointcloud: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
        pitch: float = 0.02,
        name: str = "world_pc",
        pose: List[float] | None = None,
        filter_close_points: float = 0.0,
    ) -> "Mesh":
        """Create a deterministic voxel-surface mesh from a point cloud.

        This is a deliberately dense, dependency-free replacement for the
        trimesh voxelisation path.  It creates boundary quads for occupied
        voxels, triangulates each quad consistently, and is useful for
        mapper/collision export on CPU and MPS.  Surface extraction is
        discrete, therefore the returned topology is not differentiable with
        respect to input point positions.
        """
        if pitch <= 0:
            raise ValueError("pitch must be positive")
        points = torch.as_tensor(pointcloud, dtype=torch.float64).reshape(-1, 3)
        if filter_close_points < 0:
            raise ValueError("filter_close_points must be nonnegative")
        if filter_close_points:
            points = points[torch.linalg.vector_norm(points, dim=-1) > filter_close_points]
        if points.numel() == 0:
            return Mesh(name, pose=pose, vertices=[[0.0, 0.0, 0.0]], faces=[[0, 0, 0]])
        # Voxel connectivity is host-side discrete configuration work.  The
        # resulting mesh vertices/faces are normal portable tensors/arrays.
        points_cpu = points.detach().cpu()
        origin = points_cpu.amin(dim=0) - pitch
        cells = torch.floor((points_cpu - origin) / pitch).to(torch.long)
        occupied = {tuple(int(value) for value in row) for row in cells.tolist()}
        templates = (
            ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
            ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
            ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
            ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
        )
        directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        for cell in sorted(occupied):
            for direction, template in zip(directions, templates):
                neighbor = tuple(cell[index] + direction[index] for index in range(3))
                if neighbor in occupied:
                    continue
                offset = len(vertices)
                for corner in template:
                    coordinate = origin + pitch * torch.tensor([cell[index] + corner[index] for index in range(3)], dtype=origin.dtype)
                    vertices.append(coordinate.tolist())
                faces.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
        return Mesh(name, pose=pose, vertices=vertices, faces=faces)

    def get_mesh(self, process: bool = True) -> "Mesh":
        result = Mesh(
            self.name,
            pose=None if self.pose is None else list(self.pose),
            vertices=torch.as_tensor(self.vertices).clone(),
            faces=torch.as_tensor(self.faces).clone(),
            color=None if self.color is None else list(self.color),
            texture_id=self.texture_id,
            texture=self.texture,
            material=self.material,
            device_cfg=self.device_cfg,
        )
        return _copy_visual_fields(self, result)  # type: ignore[return-value]

    def get_cuboid(self) -> Cuboid:
        vertices = torch.as_tensor(self.vertices, dtype=torch.float32)
        low, high = vertices.amin(0), vertices.amax(0)
        center = (low + high) * 0.5
        return Cuboid(self.name, _offset_pose(self.pose, center, self.device_cfg), (high-low).tolist(), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def get_mesh_data(self, process: bool = True):
        return torch.as_tensor(self.vertices).tolist(), torch.as_tensor(self.faces, dtype=torch.long).tolist()

    def to_cpu(self):
        for field_name in ("vertices", "faces", "vertex_colors", "vertex_normals", "texture_uvs", "texture_image", "face_colors"):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                setattr(self, field_name, value.detach().cpu())
        return self

    def to_gpu(self):
        for field_name in ("vertices", "faces", "vertex_colors", "vertex_normals", "texture_uvs", "texture_image", "face_colors"):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                setattr(self, field_name, value.to(self.device_cfg.device))
        return self

    def update_material(self):
        # Material is visual metadata in the portable runtime.  Keep the method
        # as a no-op rather than requiring trimesh texture mutation.
        return self


@dataclass
class VoxelGrid(Obstacle):
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    voxel_size: float = 0.02
    feature_tensor: Optional[torch.Tensor] = None
    xyzr_tensor: Optional[torch.Tensor] = None
    feature_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.dims) != 3 or any(x <= 0 for x in self.dims):
            raise ValueError("dims must contain three positive lengths")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if self.feature_tensor is not None:
            self.feature_dtype = self.feature_tensor.dtype

    def get_grid_shape(self) -> tuple[List[int], List[float], List[float]]:
        shape = [round(x / self.voxel_size) for x in self.dims]
        return shape, [-x / 2 for x in self.dims], [x / 2 for x in self.dims]

    def create_xyzr_tensor(self, transform_to_origin: bool = False, device_cfg: DeviceCfg = DeviceCfg()) -> torch.Tensor:
        shape, _, _ = self.get_grid_shape()
        axes = [
            (torch.arange(size, device=device_cfg.device, dtype=device_cfg.dtype) + 0.5) * self.voxel_size - self.dims[index] * 0.5
            for index, size in enumerate(shape)
        ]
        points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        if transform_to_origin:
            points = Pose.from_list(self._pose_or_identity(), device_cfg).transform_points(points)
        return torch.cat((points, torch.zeros_like(points[..., :1])), dim=-1)

    def get_occupied_voxels(self, feature_threshold: float | None = None) -> torch.Tensor:
        if self.feature_tensor is None:
            raise ValueError("feature_tensor is required")
        if self.xyzr_tensor is None:
            self.xyzr_tensor = self.create_xyzr_tensor(device_cfg=self.device_cfg)
        features = self.feature_tensor.reshape(-1).to(device=self.xyzr_tensor.device, dtype=self.xyzr_tensor.dtype)
        if features.numel() != self.xyzr_tensor.shape[0]:
            raise ValueError("feature_tensor and xyzr_tensor must have the same number of voxels")
        threshold = -0.5 * self.voxel_size if feature_threshold is None else feature_threshold
        output = self.xyzr_tensor.clone()
        output[:, 3] = features
        return output[features > threshold]

    def get_cuboid(self) -> Cuboid:
        return Cuboid(self.name, self._pose_or_identity(), list(self.dims), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def clone(self) -> "VoxelGrid":
        return VoxelGrid(
            name=self.name, pose=None if self.pose is None else self.pose.copy(),
            dims=self.dims.copy(), voxel_size=self.voxel_size,
            feature_tensor=None if self.feature_tensor is None else self.feature_tensor.clone(),
            xyzr_tensor=None if self.xyzr_tensor is None else self.xyzr_tensor.clone(),
            feature_dtype=self.feature_dtype, device_cfg=self.device_cfg,
        )


@dataclass
class SceneCfg(Sequence[Obstacle]):
    sphere: Optional[List[Sphere]] = None
    cuboid: Optional[List[Cuboid]] = None
    capsule: Optional[List[Obstacle]] = None
    cylinder: Optional[List[Obstacle]] = None
    mesh: Optional[List[Mesh]] = None
    voxel: Optional[List[VoxelGrid]] = None
    objects: Optional[List[Obstacle]] = None

    def __post_init__(self) -> None:
        for name in ("sphere", "cuboid", "capsule", "cylinder", "mesh", "voxel"):
            if getattr(self, name) is None:
                setattr(self, name, [])
        if self.objects is None:
            self.objects = self.sphere + self.cuboid + self.capsule + self.mesh + self.cylinder + self.voxel

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> Obstacle:
        return self.objects[index]

    @staticmethod
    def create(data_dict: dict[str, Any]) -> "SceneCfg":
        raw = data_dict.get("world_cfg", data_dict)
        return SceneCfg(
            cuboid=[Cuboid(name=n, **v) for n, v in raw.get("cuboid", {}).items()],
            sphere=[Sphere(name=n, **v) for n, v in raw.get("sphere", {}).items()],
            capsule=[Capsule(name=n, **v) for n, v in raw.get("capsule", {}).items()],
            cylinder=[Cylinder(name=n, **v) for n, v in raw.get("cylinder", {}).items()],
            mesh=[Mesh(name=n, **v) for n, v in raw.get("mesh", {}).items()],
            voxel=[VoxelGrid(name=n, **v) for n, v in raw.get("voxel", {}).items()],
        )

    @staticmethod
    def get_scene_graph(current_world: "SceneCfg", process_color: bool = True):
        """Return an external trimesh scene graph when that optional package exists.

        Collision and planning never need trimesh, so it remains optional on
        Metal-only installations.  This method deliberately refuses to
        fabricate a Warp/trimesh scene graph; callers can use
        :meth:`save_scene_as_mesh` for a portable OBJ export instead.
        """
        try:
            import trimesh  # type: ignore[import-not-found]
        except ImportError as error:
            raise NotImplementedError(
                "trimesh scene graphs are an optional visualization dependency; "
                "use save_scene_as_mesh() for portable OBJ export"
            ) from error
        graph = trimesh.Scene(base_frame="world_origin")
        for mesh in SceneCfg.create_mesh_scene(current_world).mesh:
            try:
                visual_mesh = mesh.get_trimesh_mesh(process_color=process_color)
            except NotImplementedError as error:
                raise NotImplementedError(
                    "portable Mesh cannot create a trimesh object; use save_scene_as_mesh()"
                ) from error
            graph.add_geometry(
                visual_mesh,
                geom_name=mesh.name,
                parent_node_name="world_origin",
                transform=mesh.get_transform_matrix(),
            )
        return graph

    def clone(self) -> "SceneCfg":
        return SceneCfg(
            sphere=[item.clone() for item in self.sphere],
            cuboid=[item.clone() for item in self.cuboid],
            capsule=[item.clone() for item in self.capsule],
            cylinder=[item.clone() for item in self.cylinder],
            mesh=[item.clone() for item in self.mesh],
            voxel=[item.clone() for item in self.voxel],
        )

    @staticmethod
    def create_obb_world(current_world: "SceneCfg") -> "SceneCfg":
        if current_world.voxel:
            raise NotImplementedError("VoxelGrid cannot be converted to an OBB world")
        cuboids = list(current_world.cuboid)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cylinder, *current_world.mesh):
            cuboids.append(obstacle.get_cuboid())
        return SceneCfg(cuboid=cuboids)

    @staticmethod
    def create_mesh_scene(current_world: "SceneCfg", process: bool = False) -> "SceneCfg":
        if current_world.voxel:
            raise NotImplementedError("VoxelGrid cannot be converted to a triangle mesh world")
        meshes = list(current_world.mesh)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cuboid, *current_world.cylinder):
            meshes.append(obstacle.get_mesh(process=process))
        return SceneCfg(mesh=meshes)

    @staticmethod
    def create_collision_support_world(current_world: "SceneCfg", process: bool = True) -> "SceneCfg":
        """Convert analytic geometry to layers consumed by SceneData/SceneCollision."""
        meshes = list(current_world.mesh)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cylinder):
            meshes.append(obstacle.get_mesh(process=process))
        return SceneCfg(cuboid=list(current_world.cuboid), mesh=meshes, voxel=list(current_world.voxel))

    @staticmethod
    def create_merged_mesh_world(current_world: "SceneCfg", process: bool = True, process_color: bool = True) -> "SceneCfg":
        world = SceneCfg.create_mesh_scene(current_world, process=process)
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        for mesh in world.mesh:
            local = torch.as_tensor(mesh.vertices, **mesh.device_cfg.as_torch_dict())
            transformed = Pose.from_list(mesh._pose_or_identity(), mesh.device_cfg).transform_points(local)
            start = len(vertices)
            vertices.extend(transformed.detach().cpu().tolist())
            faces.extend((torch.as_tensor(mesh.faces, dtype=torch.long) + start).tolist())
        return SceneCfg(mesh=[Mesh("merged_mesh", pose=[0, 0, 0, 1, 0, 0, 0], vertices=vertices, faces=faces)])

    def get_obb_world(self) -> "SceneCfg":
        return self.create_obb_world(self)

    def get_mesh_world(self, merge_meshes: bool = False, process: bool = False) -> "SceneCfg":
        return self.create_merged_mesh_world(self, process=process) if merge_meshes else self.create_mesh_scene(self, process=process)

    def get_collision_check_world(self, mesh_process: bool = False) -> "SceneCfg":
        return self.create_collision_support_world(self, process=mesh_process)

    def save_scene_as_mesh(
        self,
        file_path: str,
        save_as_scene_graph: bool = False,
        process_color: bool = True,
    ) -> None:
        """Export all analytic/mesh obstacles as one deterministic OBJ file.

        OBJ is intentionally selected because it needs no external graphics
        package.  The ``save_as_scene_graph`` and ``process_color`` arguments
        are accepted for source compatibility; OBJ stores merged geometry and
        does not represent a scene graph or material textures.
        """
        if save_as_scene_graph:
            raise NotImplementedError(
                "portable OBJ export contains merged geometry; trimesh is required for scene-graph export"
            )
        merged = self.create_merged_mesh_world(self, process=not process_color)
        merged.mesh[0].save_as_mesh(file_path, transform_with_pose=True)

    def add_color(self, rgba=[0.0, 0.0, 0.0, 1.0]) -> None:
        if len(rgba) not in (3, 4):
            raise ValueError("color must be RGB or RGBA")
        for obstacle in self.objects:
            obstacle.color = list(rgba)

    def add_material(self, material=Material()) -> None:
        for obstacle in self.objects:
            obstacle.material = material

    def randomize_color(self, r=[0, 1], g=[0, 1], b=[0, 1]) -> None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        for obstacle in self.objects:
            values = [
                float(torch.empty((), dtype=torch.float32).uniform_(lo, hi, generator=generator))
                for lo, hi in (r, g, b)
            ]
            obstacle.color = values + [1.0]

    def remove_absolute_paths(self) -> None:
        for obstacle in self.objects:
            if obstacle.name.startswith("/"):
                obstacle.name = obstacle.name.lstrip("/")
            if isinstance(obstacle, Mesh) and obstacle.file_path:
                obstacle.file_path = Path(obstacle.file_path).name

    def get_cache_dict(self) -> dict[str, int]:
        # ``obb`` is the pinned public spelling; ``cuboid`` keeps compatibility
        # with portable collision cache call sites added before the audit.
        return {"obb": len(self.cuboid), "cuboid": len(self.cuboid), "mesh": len(self.mesh), "voxel": len(self.voxel)}

    def add_obstacle(self, obstacle: Obstacle) -> None:
        mapping = {
            Cuboid: self.cuboid, Sphere: self.sphere, Capsule: self.capsule,
            Cylinder: self.cylinder, Mesh: self.mesh, VoxelGrid: self.voxel,
        }
        for cls, target in mapping.items():
            if isinstance(obstacle, cls):
                target.append(obstacle)
                self.objects.append(obstacle)
                return
        raise NotImplementedError(f"unsupported obstacle type: {type(obstacle).__name__}")

    def get_obstacle(self, name: str) -> Optional[Obstacle]:
        return next((x for x in self.objects if x.name == name), None)

    def remove_obstacle(self, name: str) -> None:
        obstacle = self.get_obstacle(name)
        if obstacle is None:
            return
        self.objects.remove(obstacle)
        for collection in (
            self.sphere, self.cuboid, self.capsule, self.cylinder, self.mesh, self.voxel
        ):
            if obstacle in collection:
                collection.remove(obstacle)


WorldConfig = SceneCfg

__all__ = [
    "Capsule", "Cuboid", "Cylinder", "Material", "Mesh", "Obstacle",
    "PointCloud", "SceneCfg", "Sphere", "VoxelGrid", "WorldConfig",
    "batch_tensor_cube", "tensor_capsule", "tensor_cube", "tensor_sphere",
]
