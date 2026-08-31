"""Portable mesh-obstacle storage backed by ordinary PyTorch tensors.

The pinned cuRobo module stores Warp BVH handles in this layer.  Metal cannot
share those handles, but it can retain the same mutable per-environment layout
and expose its geometry directly to :mod:`curobo_metal.ops.world_collision`.
``mesh_ids`` are therefore stable *portable cache identifiers*, never raw Warp
IDs.  All distance work is delegated to the production vectorized triangle
operator; this file deliberately does not implement a second mesh kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.warp import init_warp, warp_support_bvh_constructor_type
from curobo_metal.ops.world_collision import Mesh as BackendMesh
from curobo_metal.ops.world_collision import MeshDistanceResult, mesh_distance

from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, pose_vector, raw_warp
from .helper_pose import get_obs_idx, load_transform_from_inv_pose

wp = None


def _rotation_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return a matrix for one ``wxyz`` quaternion without leaving its device."""
    q = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(torch.finfo(quaternion.dtype).eps)
    w, x, y, z = q.unbind()
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
        torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
        torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
    ))


@dataclass(frozen=True)
class _WarpMeshCachePortable:
    """Immutable local geometry shared by every environment using ``name``.

    ``mesh_id`` is intentionally a portable, monotonically allocated cache ID.
    ``mesh`` retains the source metadata for callers that need to reconstruct a
    serialisable :class:`~curobo._src.geom.types.Mesh`; it is not a Warp object.
    """

    name: str
    mesh_id: int
    vertices: torch.Tensor
    faces: torch.Tensor
    mesh: object = None
    watertight: bool = False

    def get_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vertices.amin(0), self.vertices.amax(0)


class MeshDataWarp(PortableWarpStruct):
    """Raw Warp structure boundary; use :class:`MeshData` query methods instead."""


@dataclass(init=False)
class _MeshDataPortable(PortableObstacleData):
    """Mutable, multi-environment mesh data with vectorized CPU/MPS queries.

    Geometry is cached once by mesh name, matching cuRobo's shared Warp cache
    ownership.  Reusing a name requires byte-identical local vertices/faces;
    this avoids silently changing active geometry in another environment.
    """

    @classmethod
    def create_cache(cls, max_n, num_envs, device_cfg, max_dist=0.1):
        if max_n < 1 or num_envs < 1:
            raise ValueError("max_n and num_envs must be positive")
        if max_dist <= 0:
            raise ValueError("max_dist must be positive")
        output = cls._base(max_n, num_envs, device_cfg)
        output.mesh_ids = torch.zeros((num_envs, max_n), dtype=torch.int64, device=device_cfg.device)
        output.dims = torch.zeros((num_envs, max_n, 4), **device_cfg.as_torch_dict())
        output._mesh_cache: dict[str, WarpMeshCache] = {}
        output.wp_cache = output._mesh_cache  # pinned public field name
        output.max_dist = float(max_dist)
        output._wp_device = None
        output._next_mesh_id = 1
        return output

    @classmethod
    def from_scene_cfg(cls, scene_cfg, device_cfg, env_idx=0, num_envs=1, max_n=None, max_dist=0.1):
        meshes = list(getattr(scene_cfg, "mesh", None) or [])
        output = cls.create_cache(max_n or max(len(meshes), 1), num_envs, device_cfg, max_dist)
        output.load_batch(meshes, env_idx)
        return output

    @classmethod
    def from_batch_scene_cfg(cls, scene_cfg_list, device_cfg, max_n=None, max_dist=0.1):
        if not scene_cfg_list:
            raise ValueError("scene_cfg_list must not be empty")
        output = cls.create_cache(
            max_n or max([len(getattr(scene, "mesh", None) or []) for scene in scene_cfg_list] + [1]),
            len(scene_cfg_list), device_cfg, max_dist,
        )
        for index, scene in enumerate(scene_cfg_list):
            output.load_batch(list(getattr(scene, "mesh", None) or []), index)
        return output

    def _mesh_tensors(self, mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
        if mesh.vertices is None or mesh.faces is None:
            raise NotImplementedError("file-backed mesh loading requires caller-supplied triangular vertices/faces")
        if isinstance(mesh.vertices, torch.Tensor) and not self.device_cfg.is_same_torch_device(mesh.vertices.device):
            raise ValueError("mesh vertices must already reside on the mesh-data device")
        if isinstance(mesh.faces, torch.Tensor) and not self.device_cfg.is_same_torch_device(mesh.faces.device):
            raise ValueError("mesh faces must already reside on the mesh-data device")
        vertices = torch.as_tensor(mesh.vertices, **self.device_cfg.as_torch_dict()).clone()
        faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device_cfg.device).clone()
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
            raise ValueError("mesh vertices must have shape [V,3], V > 0")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError("mesh faces must have shape [F,3], F > 0")
        if bool(((faces < 0) | (faces >= len(vertices))).any().item()):
            raise ValueError("mesh faces contain an out-of-range vertex")
        triangles = vertices[faces]
        area = torch.linalg.vector_norm(torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
        if bool((area <= torch.finfo(vertices.dtype).eps).any().item()):
            raise ValueError("mesh faces must be nondegenerate triangles")
        return vertices, faces

    def _cache_mesh_tensors(
        self, mesh: Mesh, vertices: torch.Tensor, faces: torch.Tensor
    ) -> WarpMeshCache:
        """Return the stable cache record after local geometry was validated."""
        cached = self._mesh_cache.get(mesh.name)
        if cached is not None:
            if not (torch.equal(cached.vertices, vertices) and torch.equal(cached.faces, faces)):
                raise ValueError(
                    f"mesh name {mesh.name!r} is already cached with different geometry; use a distinct name"
                )
            return cached
        cached = WarpMeshCache(
            mesh.name, self._next_mesh_id, vertices, faces, mesh,
            bool(getattr(mesh, "watertight", False)),
        )
        self._next_mesh_id += 1
        self._mesh_cache[mesh.name] = cached
        return cached

    def _load_mesh_into_cache(self, mesh: Mesh) -> WarpMeshCache:
        vertices, faces = self._mesh_tensors(mesh)
        return self._cache_mesh_tensors(mesh, vertices, faces)

    _load_mesh_to_warp = _load_mesh_into_cache

    def _validated_pose_vector(self, pose) -> torch.Tensor:
        """Return a finite, normalized-frame-compatible portable pose row."""
        raw = pose
        if hasattr(raw, "get_pose_vector"):
            raw = raw.get_pose_vector()
        elif hasattr(raw, "position") and hasattr(raw, "quaternion"):
            raw = torch.cat((raw.position, raw.quaternion), dim=-1)
        if isinstance(raw, torch.Tensor) and not self.device_cfg.is_same_torch_device(raw.device):
            raise ValueError("mesh pose must already reside on the mesh-data device")
        vector = pose_vector(raw, self.device_cfg)
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError("mesh pose must contain finite values")
        if bool((torch.linalg.vector_norm(vector[3:]) <= torch.finfo(vector.dtype).eps).item()):
            raise ValueError("mesh pose quaternion must have nonzero norm")
        return vector

    def _inverse_world_pose(self, pose) -> torch.Tensor:
        """Validate a finite nonzero quaternion before storing its inverse."""
        vector = self._validated_pose_vector(pose)
        return inverse_pose(vector, self.device_cfg)

    def _clear_environment_buffers(self, env_idx: int) -> None:
        """Reset every fixed slot, not merely the enable bit, before a load."""
        self.mesh_ids[env_idx].zero_()
        self.dims[env_idx].zero_()
        self.inv_pose[env_idx].zero_()
        self.inv_pose[env_idx, :, 3] = 1

    def load_batch(self, meshes: Sequence[Mesh], env_idx: int):
        self._check_env(env_idx)
        meshes = list(meshes)
        if len(meshes) > self.max_n:
            raise ValueError("mesh cache capacity exceeded")
        if any(not isinstance(mesh, Mesh) for mesh in meshes):
            raise TypeError("meshes must contain curobo Mesh values")
        names = [mesh.name for mesh in meshes]
        if len(set(names)) != len(names):
            raise ValueError("mesh names must be unique within one environment")

        # Finish every fallible conversion before replacing the live
        # environment. A malformed mesh must not leave a partially updated
        # world visible to an active collision checker.
        prepared = []
        for mesh in meshes:
            vertices, faces = self._mesh_tensors(mesh)
            pose = mesh.pose if mesh.pose is not None else [0, 0, 0, 1, 0, 0, 0]
            prepared.append((mesh, vertices, faces, self._inverse_world_pose(pose)))

        cached = [self._cache_mesh_tensors(mesh, vertices, faces) for mesh, vertices, faces, _ in prepared]
        self.clear(env_idx)
        self._clear_environment_buffers(env_idx)
        for index, ((mesh, _, _, inverse), entry) in enumerate(zip(prepared, cached)):
            self.names[env_idx][index] = mesh.name
            self.mesh_ids[env_idx, index] = entry.mesh_id
            low, high = entry.get_bounds()
            self.dims[env_idx, index, :3] = high - low
            self.inv_pose[env_idx, index, :7] = inverse
            self.enable[env_idx, index] = 1
        self.count[env_idx] = len(prepared)

    def add(self, mesh: Mesh, env_idx=0):
        self._check_env(env_idx)
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a curobo Mesh")
        if self.has_name(mesh.name, env_idx):
            raise ValueError(f"mesh {mesh.name!r} already exists in environment {env_idx}")
        index = self.get_active_count(env_idx)
        if index >= self.max_n:
            raise ValueError("mesh cache capacity exceeded")
        pose = mesh.pose if mesh.pose is not None else [0, 0, 0, 1, 0, 0, 0]
        inverse = self._inverse_world_pose(pose)
        cached = self._load_mesh_into_cache(mesh)
        self.names[env_idx][index] = mesh.name
        self.mesh_ids[env_idx, index] = cached.mesh_id
        low, high = cached.get_bounds()
        self.dims[env_idx, index, :3] = high - low
        self.inv_pose[env_idx, index, :7] = inverse
        self.enable[env_idx, index] = 1
        self.count[env_idx] += 1
        return index

    def update_mesh(self, mesh: Mesh, env_idx: int = 0) -> int:
        """Update pose/metadata of a cached mesh without accepting a Warp ID.

        The cache is intentionally immutable geometry.  Replacing triangles
        under an active shared name is rejected so another environment cannot
        observe an unannounced mutation; use ``clear(..., clear_warp_cache=True)``
        after deactivating all users, or a distinct mesh name.
        """
        self._check_env(env_idx)
        if not self.has_name(mesh.name, env_idx):
            return self.add(mesh, env_idx)
        pose = mesh.pose if mesh.pose is not None else [0, 0, 0, 1, 0, 0, 0]
        inverse = self._inverse_world_pose(pose)
        cached = self._load_mesh_into_cache(mesh)
        index = self.get_idx(mesh.name, env_idx)
        self.mesh_ids[env_idx, index] = cached.mesh_id
        low, high = cached.get_bounds()
        self.dims[env_idx, index, :3] = high - low
        self.inv_pose[env_idx, index, :7] = inverse
        self.enable[env_idx, index] = 1
        return index

    def update_pose(self, name, w_obj_pose=None, obj_w_pose=None, env_idx=0):
        self._check_env(env_idx)
        index = self.get_idx(name, env_idx)
        if w_obj_pose is not None:
            value = self._inverse_world_pose(w_obj_pose)
        elif obj_w_pose is not None:
            value = self._validated_pose_vector(obj_w_pose)
        else:
            raise ValueError("w_obj_pose or obj_w_pose is required")
        self.inv_pose[env_idx, index, :7] = value

    def update_from_warp_id(self, warp_mesh_id, name, w_obj_pose=None, obj_w_pose=None, env_idx=0, mesh_idx=None):
        raise NotImplementedError(
            "raw Warp mesh IDs are unavailable on the portable backend; use update_mesh(Mesh(...))"
        )

    def get_cached_mesh_names(self) -> list[str]:
        return list(self._mesh_cache)

    def get_bounds(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._mesh_cache[name].get_bounds()
        except KeyError as error:
            raise ValueError(f"mesh {name!r} is not present in the shared cache") from error

    def get_world_pose(self, name: str, env_idx: int = 0) -> torch.Tensor:
        """Return the current world-to-object inverse buffer inverted to world pose."""
        self._check_env(env_idx)
        return inverse_pose(self.inv_pose[env_idx, self.get_idx(name, env_idx), :7], self.device_cfg)

    def get_meshes(self, env_idx: int = 0, *, include_disabled: bool = False) -> list[Mesh]:
        """Reconstruct local meshes with their current pose for scene interop.

        ``Mesh`` is serialisable and therefore receives detached CPU metadata;
        use :meth:`as_backend_meshes` or :meth:`query_points` to retain tensor
        device residency and differentiability for query points.
        """
        self._check_env(env_idx)
        result: list[Mesh] = []
        for index, name in enumerate(self.names[env_idx]):
            if name is None or (not include_disabled and not bool(self.enable[env_idx, index].item())):
                continue
            cached = self._mesh_cache[name]
            pose = self.get_world_pose(name, env_idx).detach().cpu().tolist()
            source = cached.mesh
            metadata = {"device_cfg": self.device_cfg}
            if getattr(source, "color", None) is not None:
                metadata["color"] = source.color
            if getattr(source, "material", None) is not None:
                metadata["material"] = source.material
            result.append(Mesh(
                name, pose=pose, vertices=cached.vertices.detach().cpu().tolist(),
                faces=cached.faces.detach().cpu().tolist(), **metadata,
            ))
        return result

    def as_scene_cfg(self, env_idx: int = 0, *, include_disabled: bool = False) -> SceneCfg:
        """Return an in-memory scene suitable for :class:`SceneCollision` loading."""
        return SceneCfg(mesh=self.get_meshes(env_idx, include_disabled=include_disabled))

    def as_backend_meshes(self) -> tuple[BackendMesh, ...]:
        """Return cached local triangle records for production distance queries."""
        return tuple(BackendMesh(entry.vertices, entry.faces, entry.watertight) for entry in self._mesh_cache.values())

    def query_points(
        self, points: torch.Tensor, *, env_indices: torch.Tensor | None = None, signed: bool = False,
    ) -> MeshDistanceResult:
        """Compute selected mesh distance/gradient through the production CPU/MPS operator.

        ``points`` follows :func:`curobo_metal.ops.world_collision.mesh_distance`
        (``[Q,3]`` or ``[B,Q,3]``); environment rows retain this cache's current
        poses and enable bits.  Signed distance requires every selected cached
        mesh to be explicitly declared watertight by a caller-provided mesh
        object, just like the production operator.
        """
        if not self._mesh_cache:
            raise ValueError("cannot query an empty mesh cache")
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if not self.device_cfg.is_same_torch_device(points.device):
            raise ValueError(f"points must be on {self.device_cfg.device}")
        if points.dtype != self.device_cfg.dtype:
            raise TypeError(f"points must have dtype {self.device_cfg.dtype}")
        entries = tuple(self._mesh_cache.values())
        translations = torch.zeros((self.num_envs, len(entries), 3), **self.device_cfg.as_torch_dict())
        rotations = torch.eye(3, **self.device_cfg.as_torch_dict()).expand(self.num_envs, len(entries), 3, 3).clone()
        active = torch.zeros((self.num_envs, len(entries)), dtype=torch.bool, device=self.device_cfg.device)
        index_for_name = {entry.name: index for index, entry in enumerate(entries)}
        for environment in range(self.num_envs):
            for slot, name in enumerate(self.names[environment]):
                if name is None:
                    continue
                entry_index = index_for_name[name]
                pose = self.get_world_pose(name, environment)
                translations[environment, entry_index] = pose[:3]
                rotations[environment, entry_index] = _rotation_from_wxyz(pose[3:])
                active[environment, entry_index] = self.enable[environment, slot].bool()
        return mesh_distance(
            points, self.as_backend_meshes(), translations, rotations,
            env_mesh_active=active, env_indices=env_indices, signed=signed,
        )

    def clear(self, env_idx=None, clear_warp_cache=False):
        targets = range(self.num_envs) if env_idx is None else (env_idx,)
        for index in targets:
            self._check_env(index)
        if clear_warp_cache:
            target_set = set(targets)
            still_referenced = [
                name
                for index, names in enumerate(self.names)
                if index not in target_set
                for name in names
                if name is not None
            ]
            if still_referenced:
                raise ValueError(
                    "cannot clear the shared mesh cache while another environment still references it"
                )
        super().clear(env_idx)
        for index in targets:
            self._clear_environment_buffers(index)
        if clear_warp_cache:
            self._mesh_cache.clear()
            self._next_mesh_id = 1


def is_obs_enabled(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32) -> wp.bool: raise NotImplementedError
def load_obstacle_transform(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32) -> wp.transform: raise NotImplementedError
def compute_local_sdf(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32, local_pt: wp.vec3) -> wp.float32: raise NotImplementedError
def compute_local_sdf_with_grad(obs_set: MeshDataWarp, env_idx: wp.int32, local_idx: wp.int32, local_pt: wp.vec3, query_distance: wp.float32) -> wp.vec4: raise NotImplementedError


@dataclass(frozen=True)
class WarpMeshCache:
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError


class MeshData:
    @classmethod
    def create_cache(cls, max_n: int, num_envs: int, device_cfg: DeviceCfg, max_dist: float = 0.1) -> MeshData: raise NotImplementedError
    @classmethod
    def from_scene_cfg(cls, scene_cfg: SceneCfg, device_cfg: DeviceCfg, env_idx: int = 0, num_envs: int = 1, max_n: Optional[int] = None, max_dist: float = 0.1) -> MeshData: raise NotImplementedError
    @classmethod
    def from_batch_scene_cfg(cls, scene_cfg_list: List[SceneCfg], device_cfg: DeviceCfg, max_n: Optional[int] = None, max_dist: float = 0.1) -> MeshData: raise NotImplementedError
    def load_batch(self, meshes: List[Mesh], env_idx: int) -> None: raise NotImplementedError
    def add(self, mesh: Mesh, env_idx: int = 0) -> int: raise NotImplementedError
    def update_pose(self, name: str, w_obj_pose: Optional[Pose] = None, obj_w_pose: Optional[Pose] = None, env_idx: int = 0) -> None: raise NotImplementedError
    def update_from_warp_id(self, warp_mesh_id: int, name: str, w_obj_pose: Optional[Pose] = None, obj_w_pose: Optional[Pose] = None, env_idx: int = 0, mesh_idx: Optional[int] = None) -> None: raise NotImplementedError
    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None: raise NotImplementedError
    def has_name(self, name: str, env_idx: int = 0) -> bool: raise NotImplementedError
    def get_idx(self, name: str, env_idx: int = 0) -> int: raise NotImplementedError
    def get_active_count(self, env_idx: int = 0) -> int: raise NotImplementedError
    def get_names(self, env_idx: int = 0) -> List[str]: raise NotImplementedError
    def get_cached_mesh_names(self) -> List[str]: raise NotImplementedError
    def clear(self, env_idx: Optional[int] = None, clear_warp_cache: bool = False) -> None: raise NotImplementedError
    def to_warp(self, max_dist: Optional[float] = None) -> MeshDataWarp: raise NotImplementedError


if not TYPE_CHECKING:
    WarpMeshCache = _WarpMeshCachePortable
    MeshData = _MeshDataPortable

__all__ = [
    "MeshData", "MeshDataWarp", "WarpMeshCache", "is_obs_enabled", "load_obstacle_transform",
    "compute_local_sdf", "compute_local_sdf_with_grad",
]
