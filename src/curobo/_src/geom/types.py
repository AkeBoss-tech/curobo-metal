"""Pinned cuRobo geometry records backed by portable collision data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence

import torch

from curobo._src.types.device_cfg import DeviceCfg


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


@dataclass
class Cuboid(Obstacle):
    dims: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pose is None:
            raise ValueError("Cuboid Obstacle requires Pose")
        if len(self.dims) != 3 or any(x < 0 for x in self.dims):
            raise ValueError("dims must contain three nonnegative lengths")


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
            mesh=[Mesh(name=n, **v) for n, v in raw.get("mesh", {}).items()],
            voxel=[VoxelGrid(name=n, **v) for n, v in raw.get("voxel", {}).items()],
        )

    def clone(self) -> "SceneCfg":
        return SceneCfg(
            sphere=self.sphere.copy(), cuboid=self.cuboid.copy(),
            capsule=self.capsule.copy(), cylinder=self.cylinder.copy(),
            mesh=self.mesh.copy(), voxel=self.voxel.copy(),
        )

    def get_cache_dict(self) -> dict[str, int]:
        return {"cuboid": len(self.cuboid), "mesh": len(self.mesh), "voxel": len(self.voxel)}

    def add_obstacle(self, obstacle: Obstacle) -> None:
        mapping = {Cuboid: self.cuboid, Sphere: self.sphere, Mesh: self.mesh, VoxelGrid: self.voxel}
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
        for collection in (self.sphere, self.cuboid, self.mesh, self.voxel):
            if obstacle in collection:
                collection.remove(obstacle)


WorldConfig = SceneCfg

__all__ = ["Cuboid", "Material", "Mesh", "Obstacle", "SceneCfg", "Sphere", "VoxelGrid", "WorldConfig"]
