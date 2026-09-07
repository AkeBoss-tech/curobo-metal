"""Pinned cuRobo geometry records backed by portable collision data."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import zlib
from pathlib import Path
import struct
from typing import Any as scene
from typing import Any as trimesh
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.camera import CameraObservation
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.mesh_triangulation import triangulate_mesh_faces
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import get_assets_path, join_path

class _ArrayTextureImage:
    def __init__(self, image: Any) -> None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        self._image = np.asarray(image)

    def convert(self, mode: str) -> np.ndarray:
        if mode != "RGBA":
            return self._image.copy()
        if self._image.shape[-1] == 4:
            return self._image.copy()
        alpha = np.full((*self._image.shape[:-1], 1), 255, dtype=self._image.dtype)
        return np.concatenate((self._image, alpha), axis=-1)


class _TextureMaterial:
    def __init__(self, image: Any) -> None:
        self.image = _ArrayTextureImage(image)


class TextureVisuals:
    def __init__(self, image: Any) -> None:
        self.material = _TextureMaterial(image)


def _portable_tensor(value: Any, device_cfg: DeviceCfg) -> torch.Tensor:
    """Convert a geometry value while preserving an existing tensor's graph."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device_cfg.device, dtype=device_cfg.dtype)
    return torch.as_tensor(value, device=device_cfg.device, dtype=device_cfg.dtype)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_pose_tensor(pose: torch.Tensor, *, name: str = "pose") -> None:
    if pose.shape[-1:] != (7,):
        raise ValueError(f"{name} must end in seven xyz+wxyz values")
    _require_finite(name, pose)
    if bool((torch.linalg.vector_norm(pose[..., 3:], dim=-1) <= torch.finfo(pose.dtype).eps).any().item()):
        raise ValueError(f"{name} quaternion must be nonzero")


def _batched_radius(value: Any, device_cfg: DeviceCfg) -> torch.Tensor:
    """Normalize scalar, ``[B]``, and ``[B,1]`` radius storage."""
    radius = _portable_tensor(value, device_cfg)
    if radius.ndim and radius.shape[-1] == 1:
        radius = radius.squeeze(-1)
    _require_finite("radius", radius)
    if bool((radius < 0).any().item()):
        raise ValueError("radius must be nonnegative")
    return radius


def _geometry_output(value: torch.Tensor, tensor: Optional[torch.Tensor], *, width: int) -> torch.Tensor:
    """Return a value or update a caller-owned geometry buffer exactly."""
    if tensor is None:
        return value
    if tensor.shape != value.shape or tensor.shape[-1:] != (width,):
        raise ValueError(f"geometry output tensor must have shape {tuple(value.shape)}")
    tensor.copy_(value.to(device=tensor.device, dtype=tensor.dtype))
    return tensor


def _serializable(value: Any) -> Any:
    """Convert portable value-model fields into JSON-safe nested data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Material):
        return {"metallic": value.metallic, "roughness": value.roughness}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


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
    pt: Union[List[float], np.array, torch.Tensor],
    radius: float,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Return portable ``[x, y, z, radius]`` sphere storage."""
    point = _portable_tensor(pt, device_cfg)
    if point.shape[-1:] != (3,):
        raise ValueError("sphere point must end in dimension 3")
    _require_finite("sphere point", point)
    radius_t = _batched_radius(radius, device_cfg)
    batch = torch.broadcast_shapes(point.shape[:-1], radius_t.shape)
    point = point.expand(*batch, 3)
    value = torch.cat((point, radius_t.expand(batch).unsqueeze(-1)), dim=-1)
    return _geometry_output(value, tensor, width=4)


def tensor_capsule(
    base: Union[List[float], torch.Tensor, np.array],
    tip: Union[List[float], torch.Tensor, np.array],
    radius: float,
    tensor: Optional[torch.Tensor] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> torch.Tensor:
    """Return portable ``[base_xyz, tip_xyz, radius]`` capsule storage."""
    base_t, tip_t = _portable_tensor(base, device_cfg), _portable_tensor(tip, device_cfg)
    if base_t.shape[-1:] != (3,) or tip_t.shape[-1:] != (3,):
        raise ValueError("capsule base and tip must end in dimension 3")
    _require_finite("capsule base", base_t)
    _require_finite("capsule tip", tip_t)
    radius_t = _batched_radius(radius, device_cfg)
    batch = torch.broadcast_shapes(base_t.shape[:-1], tip_t.shape[:-1], radius_t.shape)
    value = torch.cat((
        base_t.expand(*batch, 3), tip_t.expand(*batch, 3), radius_t.expand(batch).unsqueeze(-1),
    ), dim=-1)
    return _geometry_output(value, tensor, width=7)


def tensor_cube(
    pose: List[float],
    dims: List[float],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> List[torch.Tensor, torch.Tensor]:
    """Return ``[dimensions, inverse_pose]`` for one centered cuboid."""
    pose_t = _portable_tensor(pose, device_cfg)
    dims_t = _portable_tensor(dims, device_cfg)
    if pose_t.shape != (7,) or dims_t.shape != (3,):
        raise ValueError("cube pose and dims must have shapes [7] and [3]")
    _validate_pose_tensor(pose_t)
    _require_finite("cube dims", dims_t)
    if bool((dims_t < 0).any().item()):
        raise ValueError("cube dims must be nonnegative")
    forward = Pose(pose_t[:3], pose_t[3:], normalize_rotation=True)
    return [dims_t, forward.inverse().get_pose_vector().reshape(1, 7)]


def batch_tensor_cube(
    pose: List[List[float]],
    dims: List[List[float]],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> List[torch.Tensor]:
    """Vectorized ``tensor_cube`` for ``[batch, 7]`` cuboid poses."""
    pose_t = _portable_tensor(pose, device_cfg)
    dims_t = _portable_tensor(dims, device_cfg)
    if pose_t.ndim != 2 or pose_t.shape[-1] != 7 or dims_t.shape != (pose_t.shape[0], 3):
        raise ValueError("cube batches require pose [B,7] and dims [B,3]")
    _validate_pose_tensor(pose_t)
    _require_finite("cube dims", dims_t)
    if bool((dims_t < 0).any().item()):
        raise ValueError("cube dims must be nonnegative")
    forward = Pose(pose_t[:, :3], pose_t[:, 3:], normalize_rotation=True)
    return [dims_t, forward.inverse().get_pose_vector()]


@dataclass
class Material:
    metallic: float = 0.0
    roughness: float = 0.4

    def __post_init__(self) -> None:
        values = torch.as_tensor((self.metallic, self.roughness), dtype=torch.float64)
        _require_finite("material", values)


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
        if self.pose is not None:
            if len(self.pose) != 7:
                raise ValueError("pose must be [x, y, z, qw, qx, qy, qz]")
            _validate_pose_tensor(torch.as_tensor(self.pose, dtype=torch.float64))

    def _pose_or_identity(self) -> list[float]:
        return self.pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def _portable_clone(self) -> "Obstacle":
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

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        """Return a dependency-free object with the trimesh data protocol.

        Portable geometry algorithms consume only vertices, triangle faces,
        bounds, volume, and watertightness.  The native ``Mesh`` record now
        supplies that protocol, so callers do not need the optional trimesh
        package merely to fit spheres or inspect primitive geometry.
        """
        del process_color
        mesh = self.get_mesh(process=process)
        if transform_with_pose:
            vertices = Pose.from_list(mesh._pose_or_identity(), mesh.device_cfg).transform_points(
                torch.as_tensor(mesh.vertices, **mesh.device_cfg.as_torch_dict())
            ).reshape(-1, 3)
            mesh.vertices = vertices
            mesh.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if mesh.texture_uvs is not None and mesh.texture_image is not None:
            mesh.visual = TextureVisuals(mesh.texture_image)
        # ``trimesh.Trimesh`` exposes NumPy arrays for these public fields.
        # The portable mesh object implements the same data protocol so
        # downstream callers can use ``tolist`` and ``reshape`` unchanged.
        mesh.vertices = torch.as_tensor(mesh.vertices).detach().cpu().numpy()
        mesh.faces = torch.as_tensor(mesh.faces, dtype=torch.int64).detach().cpu().numpy()
        return mesh

    def save_as_mesh(self, file_path: str, transform_with_pose: bool = False):
        """Write an explicit OBJ mesh without a trimesh dependency.

        This only supports meshes representable by the portable primitive
        tessellators; USD/texture/scene graph export remains external.
        """
        mesh = self.get_mesh()
        if Path(file_path).suffix.lower() == ".glb" and mesh.texture_uvs is not None and mesh.texture_image is not None:
            gltf = {
                "asset": {"version": "2.0", "generator": "curobo-metal"},
                "buffers": [{"byteLength": 0}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 0}],
                "images": [{"bufferView": 0, "mimeType": "image/png"}],
                "meshes": [{"primitives": [{
                    "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
                }]}],
            }
            payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
            payload += b" " * ((4 - len(payload) % 4) % 4)
            blob = struct.pack("<4sII", b"glTF", 2, 20 + len(payload))
            blob += struct.pack("<I4s", len(payload), b"JSON") + payload
            Path(file_path).write_bytes(blob)
            return
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

    def get_mesh(self, process: bool = True) -> Mesh:
        raise NotImplementedError(f"{type(self).__name__} cannot be converted to a portable triangle mesh")

    def get_cuboid(self) -> Cuboid:
        raise NotImplementedError(f"{type(self).__name__} has no portable cuboid approximation")

    def get_sphere(self, n: int = 1) -> Sphere:
        if n != 1:
            raise NotImplementedError("portable obstacle bounding-sphere conversion currently supports one sphere")
        cuboid = self.get_cuboid()
        return Sphere(name=f"{self.name}_sphere", pose=cuboid.pose, radius=min(cuboid.dims), device_cfg=self.device_cfg)

    def get_bounding_spheres(
        self,
        num_spheres: Optional[int] = None,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        pre_transform_pose: Optional[Pose] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> List[Sphere]:
        if surface_radius < 0:
            raise ValueError("surface_radius must be nonnegative")
        if num_spheres not in (None, 1):
            from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
            from curobo._src.geom.sphere_fit.types import SphereFitType

            result = fit_spheres_to_mesh(
                self.get_mesh(process=False),
                num_spheres=num_spheres,
                surface_radius=surface_radius,
                fit_type=SphereFitType.MORPHIT if fit_type is None else fit_type,
                device_cfg=device_cfg,
            )
            centers = Pose.from_list(self._pose_or_identity(), device_cfg).transform_points(
                result.centers
            ).reshape(-1, 3)
            if pre_transform_pose is not None:
                centers = pre_transform_pose.transform_points(centers).reshape(-1, 3)
            return [
                Sphere(
                    name=f"{self.name}_sphere_{index}",
                    pose=[*center.detach().cpu().tolist(), 1.0, 0.0, 0.0, 0.0],
                    radius=float(radius.detach().cpu()),
                    device_cfg=device_cfg,
                )
                for index, (center, radius) in enumerate(zip(centers, result.radii))
            ]
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
        # ``Obstacle.scale`` precedes ``dims`` in inherited dataclass field
        # order.  Historical cuRobo call sites nevertheless commonly use
        # ``Cuboid(name, pose, dims)``.  Cuboids do not have independent mesh
        # scaling semantics, so treat that otherwise-ambiguous legacy slot as
        # dimensions when explicit ``dims`` were not supplied.
        if self.scale is not None and torch.equal(torch.as_tensor(self.dims), torch.zeros(3, dtype=torch.as_tensor(self.dims).dtype)):
            self.dims, self.scale = self.scale, None
        super().__post_init__()
        if self.pose is None:
            raise ValueError("Cuboid Obstacle requires Pose")
        dims = torch.as_tensor(self.dims)
        if dims.shape != (3,) or not bool(torch.isfinite(dims).all().item()) or bool((dims < 0).any().item()):
            raise ValueError("dims must contain three nonnegative lengths")

    def _portable_get_cuboid(self) -> "Cuboid":
        return Cuboid(self.name, list(self.pose), list(self.dims), color=self.color, material=self.material, device_cfg=self.device_cfg)

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
        half = torch.as_tensor(self.dims, dtype=torch.float32) * 0.5
        signs = torch.tensor([
            [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
            [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1],
        ], dtype=torch.float32)
        faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                 [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        return _primitive_mesh(signs * half, torch.tensor(faces, dtype=torch.long), self)

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)


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
        radius = torch.as_tensor(self.radius)
        if radius.numel() != 1 or not bool(torch.isfinite(radius).all().item()) or bool((radius < 0).any().item()):
            raise ValueError("radius must be nonnegative")
        if self.pose is not None:
            self.position = list(self.pose[:3])

    def get_cuboid(self) -> Cuboid:
        pose = [*self._pose_or_identity()[:3], 1.0, 0.0, 0.0, 0.0]
        return Cuboid(self.name, pose, dims=[2*self.radius]*3, color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
        vertices, faces = _sphere_surface(self.radius)
        return _primitive_mesh(vertices, faces, self)

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)


@dataclass
class Capsule(Obstacle):
    radius: float = 0.0
    base: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tip: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        super().__post_init__()
        radius = torch.as_tensor(self.radius)
        base, tip = torch.as_tensor(self.base), torch.as_tensor(self.tip)
        if (radius.numel() != 1 or not bool(torch.isfinite(radius).all().item())
                or bool((radius < 0).any().item()) or base.shape != (3,) or tip.shape != (3,)
                or not bool(torch.isfinite(base).all().item()) or not bool(torch.isfinite(tip).all().item())):
            raise ValueError("Capsule requires a nonnegative radius and 3D base/tip")

    def _portable_get_cuboid(self) -> Cuboid:
        base, tip = torch.tensor(self.base), torch.tensor(self.tip)
        lower, upper = torch.minimum(base, tip) - self.radius, torch.maximum(base, tip) + self.radius
        center = (lower + upper) * 0.5
        return Cuboid(self.name, _offset_pose(self.pose, center, self.device_cfg), (upper-lower).tolist(), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
        base = torch.as_tensor(self.base, dtype=torch.float64)
        tip = torch.as_tensor(self.tip, dtype=torch.float64)
        if not bool(torch.allclose(base[:2], tip[:2])):
            raise ValueError("portable capsule mesh requires an axis parallel to z")
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

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)


@dataclass
class Cylinder(Obstacle):
    radius: float = 0.0
    height: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        values = torch.as_tensor((self.radius, self.height))
        if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
            raise ValueError("Cylinder radius and height must be nonnegative")

    def _portable_get_cuboid(self) -> Cuboid:
        return Cuboid(self.name, self._pose_or_identity(), [2*self.radius, 2*self.radius, self.height],
                      color=self.color, material=self.material, device_cfg=self.device_cfg)

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
        vertices, faces = _cylinder_surface(self.radius, self.height)
        return _primitive_mesh(vertices, faces, self)

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)


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
        if not points.is_floating_point():
            points = points.to(torch.get_default_dtype())
            self.points = points
        if not bool(torch.isfinite(points).all().item()):
            raise ValueError("points must contain only finite values")
        if self.scale is not None:
            dtype = points.dtype if points.is_floating_point() else torch.get_default_dtype()
            scale = torch.as_tensor(self.scale, dtype=dtype, device=points.device)
            if scale.shape not in (torch.Size([]), torch.Size([3])):
                raise ValueError("point-cloud scale must be scalar or contain three values")
            self.points = points.to(dtype=dtype) * scale
            self.scale = None

    def get_mesh_data(self, process: bool = True) -> Tuple[List[List[float]], List[int]]:
        mesh = Mesh.from_pointcloud(torch.as_tensor(self.points).reshape(-1, 3), name=self.name, pose=self._pose_or_identity())
        return mesh.get_mesh_data(process)

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
        mesh = Mesh.from_pointcloud(
            torch.as_tensor(self.points).reshape(-1, 3),
            name=self.name,
            pose=self._pose_or_identity(),
        )
        mesh.color = None if self.color is None else list(self.color)
        mesh.material = self.material
        mesh.device_cfg = self.device_cfg
        return mesh

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)

    @staticmethod
    def from_camera_observation(
        camera_obs: CameraObservation,
        name: str = "pc_obstacle",
        pose: Optional[List[float]] = None,
    ) -> PointCloud:
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
        if vertices.ndim == 1 and vertices.numel() == 0:
            vertices = vertices.reshape(0, 3).to(torch.get_default_dtype())
            self.vertices = vertices
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [N, 3]")
        if not vertices.is_floating_point():
            vertices = vertices.to(torch.get_default_dtype())
            self.vertices = vertices
        if not bool(torch.isfinite(vertices).all().item()):
            raise ValueError("vertices must be finite values")
        if faces.ndim == 1 and faces.numel() % 3 == 0:
            faces = faces.reshape(-1, 3)
            self.faces = faces
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError("portable world collision supports triangulated faces [F, 3]")
        if faces.is_floating_point() and not bool(torch.equal(faces, torch.round(faces))):
            raise ValueError("mesh face indices must be integral")
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

    @property
    def _portable_bounds(self) -> np.ndarray:
        vertices = torch.as_tensor(self.vertices).detach().cpu().numpy()
        return np.stack((vertices.min(axis=0), vertices.max(axis=0)))

    @property
    def _portable_is_watertight(self) -> bool:
        faces = torch.as_tensor(self.faces, dtype=torch.int64).detach().cpu().tolist()
        edges: dict[tuple[int, int], int] = {}
        for a, b, c in faces:
            for start, end in ((a, b), (b, c), (c, a)):
                key = (start, end) if start < end else (end, start)
                edges[key] = edges.get(key, 0) + 1
        return bool(edges) and all(count == 2 for count in edges.values())

    @property
    def _portable_volume(self) -> float:
        vertices = torch.as_tensor(self.vertices, dtype=torch.float64)
        faces = torch.as_tensor(self.faces, dtype=torch.int64)
        triangles = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3)
        signed = torch.sum(
            triangles[:, 0] * torch.linalg.cross(triangles[:, 1], triangles[:, 2]), dim=-1
        ) / 6.0
        return float(torch.abs(signed.sum()).detach().cpu())

    @classmethod
    def from_polygon_faces(
        cls,
        name: str,
        vertices: List[List[float]],
        faces: List[int],
        face_counts: List[int],
        pose: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        color: Optional[List[float]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs: Any,
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
        pointcloud: np.ndarray,
        pitch: float = 0.02,
        name = "world_pc",
        pose: List[float] = [0, 0, 0, 1, 0, 0, 0],
        filter_close_points: float = 0.0,
    ):
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

    def _portable_get_mesh(self, process: bool = True) -> "Mesh":
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

    def _portable_get_cuboid(self) -> Cuboid:
        vertices = torch.as_tensor(self.vertices, dtype=torch.float32)
        low, high = vertices.amin(0), vertices.amax(0)
        center = (low + high) * 0.5
        return Cuboid(self.name, _offset_pose(self.pose, center, self.device_cfg), (high-low).tolist(), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def get_trimesh_mesh(
        self, process: bool = True, process_color: bool = True,
        transform_with_pose: bool = False,
    ) -> trimesh.Trimesh:
        return Obstacle.get_trimesh_mesh(self, process, process_color, transform_with_pose)

    def get_mesh_data(self, process: bool = True) -> Tuple[List[List[float]], List[int]]:
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
        dims = torch.as_tensor(self.dims)
        if dims.shape != (3,) or not bool(torch.isfinite(dims).all().item()) or bool((dims <= 0).any().item()):
            raise ValueError("dims must contain three positive lengths")
        voxel_size = torch.as_tensor(self.voxel_size)
        if voxel_size.numel() != 1 or not bool(torch.isfinite(voxel_size).all().item()) or bool((voxel_size <= 0).any().item()):
            raise ValueError("voxel_size must be positive")
        if self.feature_tensor is not None:
            if not isinstance(self.feature_tensor, torch.Tensor) or not self.feature_tensor.is_floating_point():
                raise ValueError("feature_tensor must be a floating torch.Tensor")
            if not bool(torch.isfinite(self.feature_tensor).all().item()):
                raise ValueError("feature_tensor must contain only finite values")
            self.feature_dtype = self.feature_tensor.dtype

    def get_grid_shape(self) -> Tuple[List[int], List[float], List[float]]:
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

    def get_occupied_voxels(
        self, feature_threshold: Optional[float] = None
    ) -> torch.Tensor:
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

    def _portable_get_cuboid(self) -> Cuboid:
        return Cuboid(self.name, self._pose_or_identity(), list(self.dims), color=self.color,
                      material=self.material, device_cfg=self.device_cfg)

    def clone(self) -> VoxelGrid:
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
    def create(data_dict: Dict[str, Any]) -> SceneCfg:
        """Build a scene from the pinned mapping format or JSON list form."""
        if not isinstance(data_dict, dict):
            raise TypeError("scene data must be a mapping")
        raw = data_dict.get("world_cfg", data_dict)
        if not isinstance(raw, dict):
            raise TypeError("world_cfg must be a mapping")

        def decode(kind: str, obstacle_type):
            values = raw.get(kind, {})
            if values is None:
                return []
            if isinstance(values, dict):
                entries = [{"name": name, **(fields or {})} for name, fields in values.items()]
            if isinstance(values, list):
                entries = values
            if not isinstance(values, (dict, list)):
                raise TypeError(f"{kind} must be a mapping or a list of named mappings")
            decoded = []
            for fields in entries:
                if not isinstance(fields, dict) or "name" not in fields:
                    raise ValueError(f"{kind} list entries require a name")
                fields = fields.copy()
                name = fields.pop("name")
                if isinstance(fields.get("material"), dict):
                    fields["material"] = Material(**fields["material"])
                if obstacle_type is VoxelGrid and isinstance(fields.get("feature_tensor"), list):
                    fields["feature_tensor"] = torch.as_tensor(
                        fields["feature_tensor"], dtype=torch.get_default_dtype()
                    )
                decoded.append(obstacle_type(name=name, **fields))
            return decoded

        return SceneCfg(
            cuboid=decode("cuboid", Cuboid), sphere=decode("sphere", Sphere),
            capsule=decode("capsule", Capsule), cylinder=decode("cylinder", Cylinder),
            mesh=decode("mesh", Mesh), voxel=decode("voxel", VoxelGrid),
        )

    def _portable_to_dict(self, *, wrap_world_cfg: bool = True) -> dict[str, Any]:
        """Return a deterministic JSON-safe scene mapping accepted by :meth:`create`.

        Runtime ``DeviceCfg`` objects and derived voxel ``feature_dtype`` are
        intentionally omitted: they are process-local execution settings, and
        the latter is reconstructed from ``feature_tensor`` on load.
        """
        def encode(items: Sequence[Obstacle]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for obstacle in items:
                fields = {
                    name: _serializable(value)
                    for name, value in vars(obstacle).items()
                    if name not in {"name", "device_cfg", "feature_dtype"}
                    and value is not None
                }
                # ``position`` is a deprecated Sphere alias.  Emitting both
                # would make construction prefer it and erase an otherwise
                # meaningful serialized pose quaternion.
                if isinstance(obstacle, Sphere) and obstacle.pose is not None:
                    fields.pop("position", None)
                result[obstacle.name] = fields
            return result

        payload = {
            "sphere": encode(self.sphere), "cuboid": encode(self.cuboid),
            "capsule": encode(self.capsule), "cylinder": encode(self.cylinder),
            "mesh": encode(self.mesh), "voxel": encode(self.voxel),
        }
        return {"world_cfg": payload} if wrap_world_cfg else payload

    as_dict = _portable_to_dict

    @staticmethod
    def get_scene_graph(
        current_world: SceneCfg, process_color: bool = True
    ) -> trimesh.scene.scene.Scene:
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

    def clone(self) -> SceneCfg:
        return SceneCfg(
            sphere=[item.clone() for item in self.sphere],
            cuboid=[item.clone() for item in self.cuboid],
            capsule=[item.clone() for item in self.capsule],
            cylinder=[item.clone() for item in self.cylinder],
            mesh=[item.clone() for item in self.mesh],
            voxel=[item.clone() for item in self.voxel],
        )

    @staticmethod
    def create_obb_world(current_world: SceneCfg) -> SceneCfg:
        if current_world.voxel:
            raise NotImplementedError("VoxelGrid cannot be converted to an OBB world")
        cuboids = list(current_world.cuboid)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cylinder, *current_world.mesh):
            cuboids.append(obstacle.get_cuboid())
        return SceneCfg(cuboid=cuboids)

    @staticmethod
    def create_mesh_scene(current_world: SceneCfg, process: bool = False) -> SceneCfg:
        if current_world.voxel:
            raise NotImplementedError("VoxelGrid cannot be converted to a triangle mesh world")
        meshes = list(current_world.mesh)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cuboid, *current_world.cylinder):
            meshes.append(obstacle.get_mesh(process=process))
        return SceneCfg(mesh=meshes)

    @staticmethod
    def create_collision_support_world(
        current_world: SceneCfg, process: bool = True
    ) -> SceneCfg:
        """Convert analytic geometry to layers consumed by SceneData/SceneCollision."""
        meshes = list(current_world.mesh)
        for obstacle in (*current_world.sphere, *current_world.capsule, *current_world.cylinder):
            meshes.append(obstacle.get_mesh(process=process))
        return SceneCfg(cuboid=list(current_world.cuboid), mesh=meshes, voxel=list(current_world.voxel))

    @staticmethod
    def create_merged_mesh_world(
        current_world: SceneCfg, process: bool = True, process_color: bool = True
    ) -> SceneCfg:
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

    def get_obb_world(self) -> SceneCfg:
        return self.create_obb_world(self)

    def get_mesh_world(
        self, merge_meshes: bool = False, process: bool = False
    ) -> SceneCfg:
        return self.create_merged_mesh_world(self, process=process) if merge_meshes else self.create_mesh_scene(self, process=process)

    def get_collision_check_world(self, mesh_process: bool = False) -> SceneCfg:
        return self.create_collision_support_world(self, process=mesh_process)

    def save_scene_as_mesh(
        self,
        file_path: str,
        save_as_scene_graph=False,
        process_color: bool = True,
    ):
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

    def add_color(self, rgba=[0.0, 0.0, 0.0, 1.0]):
        if len(rgba) not in (3, 4):
            raise ValueError("color must be RGB or RGBA")
        for obstacle in self.objects:
            obstacle.color = list(rgba)

    def add_material(self, material=Material()):
        for obstacle in self.objects:
            obstacle.material = material

    def randomize_color(self, r=[0, 1], g=[0, 1], b=[0, 1]):
        generator = torch.Generator(device="cpu").manual_seed(0)
        for obstacle in self.objects:
            values = [
                float(torch.empty((), dtype=torch.float32).uniform_(lo, hi, generator=generator))
                for lo, hi in (r, g, b)
            ]
            obstacle.color = values + [1.0]

    def remove_absolute_paths(self):
        for obstacle in self.objects:
            if obstacle.name.startswith("/"):
                obstacle.name = obstacle.name.lstrip("/")
            if isinstance(obstacle, Mesh) and obstacle.file_path:
                obstacle.file_path = Path(obstacle.file_path).name

    def get_cache_dict(self) -> Dict[str, int]:
        # ``obb`` is the pinned public spelling; ``cuboid`` keeps compatibility
        # with portable collision cache call sites added before the audit.
        return {"obb": len(self.cuboid), "cuboid": len(self.cuboid), "mesh": len(self.mesh), "voxel": len(self.voxel)}

    def add_obstacle(self, obstacle: Obstacle):
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

    def get_obstacle(self, name: str) -> Union[None, Obstacle]:
        return next((x for x in self.objects if x.name == name), None)

    def remove_obstacle(self, name: str):
        obstacle = self.get_obstacle(name)
        if obstacle is None:
            return
        self.objects.remove(obstacle)
        for collection in (
            self.sphere, self.cuboid, self.capsule, self.cylinder, self.mesh, self.voxel
        ):
            if obstacle in collection:
                collection.remove(obstacle)


Obstacle.clone = Obstacle._portable_clone
Cuboid.get_cuboid = Cuboid._portable_get_cuboid
Cuboid.get_mesh = Cuboid._portable_get_mesh
Sphere.get_mesh = Sphere._portable_get_mesh
Capsule.get_cuboid = Capsule._portable_get_cuboid
Capsule.get_mesh = Capsule._portable_get_mesh
Cylinder.get_cuboid = Cylinder._portable_get_cuboid
Cylinder.get_mesh = Cylinder._portable_get_mesh
PointCloud.get_mesh = PointCloud._portable_get_mesh
Mesh.bounds = Mesh._portable_bounds
Mesh.is_watertight = Mesh._portable_is_watertight
Mesh.volume = Mesh._portable_volume
Mesh.get_mesh = Mesh._portable_get_mesh
Mesh.get_cuboid = Mesh._portable_get_cuboid
VoxelGrid.get_cuboid = VoxelGrid._portable_get_cuboid
SceneCfg.to_dict = SceneCfg._portable_to_dict

WorldConfig = SceneCfg

__all__ = [
    "Capsule", "Cuboid", "Cylinder", "Material", "Mesh", "Obstacle",
    "PointCloud", "SceneCfg", "Sphere", "VoxelGrid", "WorldConfig",
    "batch_tensor_cube", "tensor_capsule", "tensor_cube", "tensor_sphere",
]
