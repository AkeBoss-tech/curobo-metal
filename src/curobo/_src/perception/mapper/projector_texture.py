"""Portable visibility-tested projective texturing.

The upstream implementation writes directly into a Warp texture atlas while
walking a CUDA sparse TSDF.  This implementation deliberately keeps the
useful part of that contract -- validated camera batches, pinhole projection,
depth visibility, deterministic best-view selection, and atlas construction
-- in ordinary PyTorch.  It runs on CPU and MPS, and does not expose a fake
Warp texture or raw CUDA launch ABI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.geom.transform import torch_quaternion_to_matrix
from curobo._src.geom.types import Mesh
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.perception.mapper.storage import BlockSparseTSDF, OccupiedVoxels
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, wp


_PROJECTIVE_TEXTURE_ATLAS_MAX_WIDTH_PX = 16_384
TextureBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _projective_texture_atlas_columns(total_cameras: int, image_width: int) -> int:
    if total_cameras <= 0:
        raise ValueError(f"total_cameras must be positive, got {total_cameras}")
    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")
    return min(total_cameras, max(1, _PROJECTIVE_TEXTURE_ATLAS_MAX_WIDTH_PX // image_width))


def _validate_texture_depth_tolerance(value: Optional[float]) -> float:
    if value is None:
        return -1.0
    value = float(value)
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("texture_depth_tolerance_m must be finite, non-negative, or None")
    return value


def _texture_depth_visible(z: torch.Tensor, depth: torch.Tensor, *, voxel_size: float,
                           texture_depth_tolerance_m: Optional[float]) -> torch.Tensor:
    valid = torch.isfinite(z) & torch.isfinite(depth) & (z > 0) & (depth > 0)
    tolerance = (torch.full_like(z, float(texture_depth_tolerance_m))
                 if texture_depth_tolerance_m is not None
                 else torch.maximum(torch.full_like(z, 2.0 * float(voxel_size)), z * 0.01))
    return valid & ((z - depth).abs() <= tolerance)


@dataclass(frozen=True)
class ProjectiveTextureProjectorCfg:
    """Construction-time RGB-D projection dimensions and depth limits."""

    texture_num_cameras: int = 1
    image_height: int = 1
    image_width: int = 1
    depth_minimum_distance: float = 0.01
    depth_maximum_distance: float = 10.0
    voxel_size: float = 0.01

    def __post_init__(self) -> None:
        if self.texture_num_cameras <= 0 or self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("texture_num_cameras, image_height, and image_width must be positive")
        if not (math.isfinite(self.depth_minimum_distance) and math.isfinite(self.depth_maximum_distance)
                and 0 <= self.depth_minimum_distance < self.depth_maximum_distance):
            raise ValueError("texture depth limits must be finite and satisfy 0 <= min < max")
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive and finite")


@dataclass(frozen=True)
class _PreparedMeshTextureProjection:
    batches: list[TextureBatch]
    texture_atlas: torch.Tensor
    atlas_columns: int
    total_cameras: int
    depth_tolerance: float


class _ProjectiveTextureProjectorPortable:
    """Project RGB/RGB-D observations onto mesh vertices or occupied voxels.

    CUDA/Warp texture-object construction and raw kernel entry points are not
    part of this class.  The public high-level operation uses exact pinhole
    geometry and nearest-pixel texture sampling on supported devices.
    """

    def __init__(self, tsdf: BlockSparseTSDF, renderer: BlockSparseTSDFRenderer,
                 config: ProjectiveTextureProjectorCfg) -> None:
        self._tsdf, self._renderer, self.config = tsdf, renderer, config

    @property
    def device(self) -> torch.device:
        state = getattr(self._tsdf, "state", None)
        if state is not None and hasattr(state, "tsdf"):
            return state.tsdf.device
        return torch.device(getattr(self._tsdf, "device"))

    def prepare_mesh_projection(self, texture_observations: CameraObservation | Sequence[CameraObservation], *,
                                texture_depth_tolerance_m: Optional[float]) -> _PreparedMeshTextureProjection:
        tolerance = _validate_texture_depth_tolerance(texture_depth_tolerance_m)
        batches = self._normalize_projective_texture_observations(texture_observations)
        total = len(batches) * self.config.texture_num_cameras
        columns = _projective_texture_atlas_columns(total, self.config.image_width)
        return _PreparedMeshTextureProjection(batches, self._build_projective_texture_atlas(batches, columns),
                                              columns, total, tolerance)

    def _as_pose_batches(self, observation: CameraObservation, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.pose is None or observation.pose.position is None or observation.pose.quaternion is None:
            raise ValueError("CameraObservation.pose is required for texture mapping")
        position, quaternion = observation.pose.position, observation.pose.quaternion
        if position.numel() != count * 3 or quaternion.numel() != count * 4:
            raise ValueError("pose batch must contain one xyz/wxyz pose per texture camera")
        return position.reshape(count, 3), quaternion.reshape(count, 4)

    def _normalize_projective_texture_batch(self, observation: CameraObservation) -> TextureBatch:
        if observation.rgb_image is None or observation.intrinsics is None:
            raise ValueError("CameraObservation.rgb_image and intrinsics are required for texture mapping")
        rgb = observation.rgb_image
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        expected_rgb = (self.config.texture_num_cameras, self.config.image_height, self.config.image_width, 3)
        if tuple(rgb.shape) != expected_rgb or rgb.dtype != torch.uint8:
            raise ValueError(f"rgb_image must be uint8 with shape {expected_rgb}, got {tuple(rgb.shape)} {rgb.dtype}")
        intrinsics = observation.intrinsics
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if tuple(intrinsics.shape) != (self.config.texture_num_cameras, 3, 3):
            raise ValueError("intrinsics must have shape [texture_num_cameras, 3, 3]")
        if not intrinsics.is_floating_point():
            raise ValueError("intrinsics must be a floating tensor")
        position, quaternion = self._as_pose_batches(observation, self.config.texture_num_cameras)
        if not position.is_floating_point() or not quaternion.is_floating_point():
            raise ValueError("camera poses must be floating tensors")
        dtype = position.dtype
        target = self.device
        rgb = rgb.to(target).contiguous()
        intrinsics = intrinsics.to(target, dtype=dtype).contiguous()
        position, quaternion = position.to(target, dtype=dtype).contiguous(), quaternion.to(target, dtype=dtype).contiguous()
        depth = observation.depth_image
        if depth is None:
            depth = self._renderer.render_depth(intrinsics, Pose(position, quaternion),
                                                (self.config.image_height, self.config.image_width))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        expected_depth = expected_rgb[:3]
        if tuple(depth.shape) != expected_depth or not depth.is_floating_point():
            raise ValueError(f"depth_image must be floating with shape {expected_depth}")
        return rgb, depth.to(target, dtype=dtype).contiguous(), intrinsics, position, quaternion

    def _stack_projective_texture_batch(self, observations: Sequence[CameraObservation]) -> TextureBatch:
        if len(observations) != self.config.texture_num_cameras:
            raise ValueError("unbatched texture observations must fill one complete camera batch")
        if any(o.rgb_image is None or o.intrinsics is None or o.pose is None for o in observations):
            raise ValueError("every texture observation needs rgb_image, intrinsics, and pose")
        rgb = torch.stack([o.rgb_image for o in observations])
        depths = [o.depth_image for o in observations]
        for depth in depths:
            if depth is not None and (
                tuple(depth.shape) != (self.config.image_height, self.config.image_width)
                or not depth.is_floating_point()
            ):
                raise ValueError(
                    "depth_image must be floating with shape "
                    f"({self.config.image_height}, {self.config.image_width})"
                )
        depth = torch.stack(depths) if all(v is not None for v in depths) else None
        position = torch.cat([o.pose.position.reshape(1, 3) for o in observations])
        quaternion = torch.cat([o.pose.quaternion.reshape(1, 4) for o in observations])
        result = self._normalize_projective_texture_batch(CameraObservation(
            rgb_image=rgb, depth_image=depth, intrinsics=torch.stack([o.intrinsics for o in observations]),
            pose=Pose(position, quaternion), depth_to_meter=1.0,
        ))
        if depth is None and any(value is not None for value in depths):
            rgb_batch, rendered, intrinsics, position, quaternion = result
            rendered = rendered.clone()
            for index, value in enumerate(depths):
                if value is not None:
                    rendered[index] = value.to(rendered.device, dtype=rendered.dtype)
            result = rgb_batch, rendered, intrinsics, position, quaternion
        return result

    def _normalize_projective_texture_observations(self, observations: CameraObservation | Sequence[CameraObservation]) -> list[TextureBatch]:
        values = [observations] if isinstance(observations, CameraObservation) else list(observations)
        if not values:
            raise ValueError("texture_observations must contain at least one CameraObservation")
        batches: list[TextureBatch] = []
        pending: list[CameraObservation] = []
        for observation in values:
            if observation.rgb_image is None:
                raise ValueError("CameraObservation.rgb_image is required for texture mapping")
            if observation.rgb_image.ndim == 4:
                if pending:
                    raise ValueError("cannot mix batched and unbatched texture observations")
                batches.append(self._normalize_projective_texture_batch(observation))
            elif observation.rgb_image.ndim == 3:
                pending.append(observation)
                if len(pending) == self.config.texture_num_cameras:
                    batches.append(self._stack_projective_texture_batch(pending))
                    pending = []
            else:
                raise ValueError("rgb_image must have shape [H,W,3] or [C,H,W,3]")
        if pending:
            raise ValueError("unbatched texture observations must form complete camera batches")
        return batches

    def _build_projective_texture_atlas(self, batches: Sequence[TextureBatch], atlas_columns: int) -> torch.Tensor:
        total = len(batches) * self.config.texture_num_cameras
        rows = math.ceil(total / atlas_columns)
        atlas = torch.zeros((rows * self.config.image_height, atlas_columns * self.config.image_width, 3),
                            dtype=torch.uint8, device=self.device)
        for batch_index, (rgb, *_rest) in enumerate(batches):
            for camera_index in range(self.config.texture_num_cameras):
                index = batch_index * self.config.texture_num_cameras + camera_index
                row, column = divmod(index, atlas_columns)
                y, x = row * self.config.image_height, column * self.config.image_width
                atlas[y:y + self.config.image_height, x:x + self.config.image_width] = rgb[camera_index]
        return atlas

    def _project_batch(self, points: torch.Tensor, batch: TextureBatch, minimum: float, maximum: float,
                       tolerance: Optional[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb, depth, intrinsics, positions, quaternions = batch
        count, point_count = rgb.shape[0], points.shape[0]
        rotations = torch_quaternion_to_matrix(quaternions)
        # world -> camera.  The source has row-vector camera rotations.
        camera_points = torch.matmul(points.to(dtype=positions.dtype).unsqueeze(0) - positions[:, None], rotations)
        z = camera_points[..., 2]
        safe_z = torch.where(z.abs() > torch.finfo(z.dtype).eps, z, torch.ones_like(z))
        u = intrinsics[:, 0, 0, None] * camera_points[..., 0] / safe_z + intrinsics[:, 0, 2, None]
        v = intrinsics[:, 1, 1, None] * camera_points[..., 1] / safe_z + intrinsics[:, 1, 2, None]
        inside = (torch.isfinite(z) & torch.isfinite(u) & torch.isfinite(v) & (z > minimum) & (z <= maximum)
                  & (u >= 0) & (u < self.config.image_width) & (v >= 0) & (v < self.config.image_height))
        px = torch.floor(u).to(torch.long).clamp(0, self.config.image_width - 1)
        py = torch.floor(v).to(torch.long).clamp(0, self.config.image_height - 1)
        camera = torch.arange(count, device=points.device)[:, None]
        visible = inside & _texture_depth_visible(z, depth[camera, py, px], voxel_size=self.config.voxel_size,
                                                   texture_depth_tolerance_m=tolerance)
        score = torch.where(visible, -z, torch.full_like(z, -torch.inf))
        best_score, best_camera = score.max(0)
        valid = torch.isfinite(best_score)
        point_index = torch.arange(point_count, device=points.device)
        color = rgb[best_camera, py[best_camera, point_index], px[best_camera, point_index]]
        return color, valid, best_score, best_camera, torch.stack((u[best_camera, point_index], v[best_camera, point_index]), -1)

    def _project_texture_points(self, points: torch.Tensor, batches: Sequence[TextureBatch], *,
                                camera_min_distance: float, camera_max_distance: float,
                                texture_depth_tolerance_m: Optional[float]) -> tuple[torch.Tensor, torch.Tensor]:
        colors = torch.zeros((points.shape[0], 3), dtype=torch.uint8, device=points.device)
        valid = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        best = torch.full((points.shape[0],), -torch.inf, dtype=points.dtype, device=points.device)
        for batch in batches:
            candidate, candidate_valid, score, _camera, _uv = self._project_batch(
                points, batch, camera_min_distance, camera_max_distance, texture_depth_tolerance_m)
            update = candidate_valid & (score > best)
            colors[update], valid[update], best[update] = candidate[update], True, score[update]
        return colors, valid

    def _project_texture_points_with_uvs(
        self, points: torch.Tensor, projection: _PreparedMeshTextureProjection, *,
        camera_min_distance: float, camera_max_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return selected colors, validity, and atlas-normalized UVs.

        This is the portable counterpart of the upstream Warp mesh UV kernel.
        Selection is per vertex (nearest visible camera, first camera on a
        tie); the raw CUDA triangle score and texture-object ABI intentionally
        remain outside the CPU/MPS contract.
        """
        point_count = int(points.shape[0])
        colors = torch.zeros((point_count, 3), dtype=torch.uint8, device=points.device)
        valid = torch.zeros(point_count, dtype=torch.bool, device=points.device)
        best = torch.full((point_count,), -torch.inf, dtype=points.dtype, device=points.device)
        selected_camera = torch.zeros(point_count, dtype=torch.long, device=points.device)
        selected_uv = torch.zeros((point_count, 2), dtype=points.dtype, device=points.device)
        tolerance = None if projection.depth_tolerance < 0 else projection.depth_tolerance
        for batch_index, batch in enumerate(projection.batches):
            candidate, candidate_valid, score, camera, image_uv = self._project_batch(
                points, batch, camera_min_distance, camera_max_distance, tolerance)
            update = candidate_valid & (score > best)
            colors[update] = candidate[update]
            valid[update] = True
            best[update] = score[update]
            selected_camera[update] = camera[update] + batch_index * self.config.texture_num_cameras
            selected_uv[update] = image_uv[update]
        atlas_width = projection.texture_atlas.shape[1]
        atlas_height = projection.texture_atlas.shape[0]
        columns = projection.atlas_columns
        tile_column = selected_camera.remainder(columns).to(points.dtype)
        tile_row = torch.div(selected_camera, columns, rounding_mode="floor").to(points.dtype)
        atlas_uv = torch.stack((
            (selected_uv[:, 0] + tile_column * self.config.image_width) / max(1, atlas_width - 1),
            (selected_uv[:, 1] + tile_row * self.config.image_height) / max(1, atlas_height - 1),
        ), -1).clamp(0, 1)
        return colors, valid, atlas_uv

    def texture_occupied_voxels(self, voxels: OccupiedVoxels,
                                texture_observations: CameraObservation | Sequence[CameraObservation], *,
                                camera_min_distance: Optional[float], camera_max_distance: Optional[float],
                                texture_depth_tolerance_m: Optional[float]) -> OccupiedVoxels:
        _validate_texture_depth_tolerance(texture_depth_tolerance_m)
        batches = self._normalize_projective_texture_observations(texture_observations)
        minimum = self.config.depth_minimum_distance if camera_min_distance is None else float(camera_min_distance)
        maximum = self.config.depth_maximum_distance if camera_max_distance is None else float(camera_max_distance)
        if not (0 <= minimum < maximum):
            raise ValueError("camera_min_distance must be non-negative and less than camera_max_distance")
        colors, valid = self._project_texture_points(voxels.centers, batches, camera_min_distance=minimum,
                                                      camera_max_distance=maximum,
                                                      texture_depth_tolerance_m=texture_depth_tolerance_m)
        return OccupiedVoxels(voxels.centers, voxels.block_idx_per_voxel, voxels.block_data,
                              texture_colors=colors, texture_valid=valid, subvoxel_factor=voxels.subvoxel_factor)

    def project_mesh(self, vertices: torch.Tensor, triangles: torch.Tensor, normals: torch.Tensor,
                     colors: torch.Tensor, projection: _PreparedMeshTextureProjection, *,
                     camera_min_distance: Optional[float], camera_max_distance: Optional[float]) -> Mesh:
        if vertices.ndim != 2 or vertices.shape[-1] != 3 or triangles.ndim != 2 or triangles.shape[-1] != 3:
            raise ValueError("vertices and triangles must be [N,3] and [M,3]")
        if normals.shape != vertices.shape or colors.shape != vertices.shape or colors.dtype != torch.uint8:
            raise ValueError("normals and uint8 colors must be parallel to vertices")
        minimum = self.config.depth_minimum_distance if camera_min_distance is None else float(camera_min_distance)
        maximum = self.config.depth_maximum_distance if camera_max_distance is None else float(camera_max_distance)
        sampled, valid, uvs = self._project_texture_points_with_uvs(
            vertices, projection, camera_min_distance=minimum, camera_max_distance=maximum)
        result_colors = torch.where(valid[:, None], sampled, colors.to(vertices.device))
        # Invalid vertices retain a deterministic zero UV and their supplied
        # fallback color.  This preserves valid mesh geometry without claiming
        # the source raw CUDA fallback-atlas patch implementation.
        uvs = torch.where(valid[:, None], uvs, torch.zeros_like(uvs))
        return Mesh(name="block_sparse_tsdf_textured_mesh", vertices=vertices, faces=triangles,
                    vertex_colors=result_colors.float() / 255.0, vertex_normals=normals,
                    texture_uvs=uvs, texture_image=projection.texture_atlas)


class ProjectiveTextureProjector:
    """Pinned cuRoboV2 declaration surface for portable texturing."""

    def __init__(
        self,
        tsdf: BlockSparseTSDF,
        renderer: BlockSparseTSDFRenderer,
        config: ProjectiveTextureProjectorCfg,
    ) -> None:
        raise NotImplementedError

    def prepare_mesh_projection(
        self,
        texture_observations: CameraObservation | Sequence[CameraObservation],
        *,
        texture_depth_tolerance_m: Optional[float],
    ) -> _PreparedMeshTextureProjection:
        raise NotImplementedError

    def project_mesh(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        normals: torch.Tensor,
        colors: torch.Tensor,
        projection: _PreparedMeshTextureProjection,
        *,
        camera_min_distance: Optional[float],
        camera_max_distance: Optional[float],
    ) -> Mesh:
        raise NotImplementedError

    def texture_occupied_voxels(
        self,
        voxels: OccupiedVoxels,
        texture_observations: CameraObservation | Sequence[CameraObservation],
        *,
        camera_min_distance: Optional[float],
        camera_max_distance: Optional[float],
        texture_depth_tolerance_m: Optional[float],
    ) -> OccupiedVoxels:
        raise NotImplementedError


# Select the full portable PyTorch implementation at runtime.  Its additional
# inspection helpers (such as ``device``) are deliberately not part of the
# pinned static declaration contract.
if not TYPE_CHECKING:
    ProjectiveTextureProjector = _ProjectiveTextureProjectorPortable


__all__ = [
    "ProjectiveTextureProjector", "ProjectiveTextureProjectorCfg", "TextureBatch",
    "_projective_texture_atlas_columns", "_texture_depth_visible", "_validate_texture_depth_tolerance",
]
