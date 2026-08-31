from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union
import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import (
    BlockDataView,
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
    MatchedVoxels,
    OccupiedVoxels,
)
from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    is_portable_dense_block_payload,
    load_block_checkpoint,
    prepare_blocks_for_import,
    save_block_checkpoint,
    validate_block_metadata_for_target,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose
from curobo._src.util.logging import deprecated, log_and_raise
from curobo_metal.ops.perception import CameraObservation as NativeObservation
from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper
from curobo_metal.ops.perception.core import DenseMap, dense_esdf
from .renderer import depth_to_colormap, normals_to_colormap

# Imported lazily by the real CUDA implementation.  Keeping the names here
# avoids a Mapper -> ESDFIntegrator -> TSDFIntegrator -> Mapper import cycle;
# the portable Mapper never consumes these raw sparse-kernel classes itself.
BlockSparseESDFIntegrator = None
BlockSparseESDFIntegratorCfg = None


def _device(value):
    if str(value).startswith("cuda"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return value


class Mapper:
    def __init__(self, config: MapperCfg):
        self.config = config
        if config.feature_dim:
            raise NotImplementedError("MapperCfg.feature_dim requires CUDA/Warp feature-volume integration")
        if config.lidar_num_sensors:
            raise NotImplementedError("MapperCfg.lidar_num_sensors requires CUDA/Warp LiDAR integration")
        center = (0.0,0.0,0.0) if config.grid_center is None else tuple(torch.as_tensor(config.grid_center).tolist())
        native = PerceptionConfig(
            config.native_grid_shape, config.voxel_size, center,
            config.truncation_distance, config.depth_minimum_distance,
            config.depth_maximum_distance, config.accumulator_w_max,
            block_size=config.block_size,
        )
        self._mapper = PerceptionMapper(native, device=_device(config.device))
        self._storage = BlockSparseTSDF.from_native(BlockSparseTSDFCfg(
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
            voxel_size=config.voxel_size,
            origin=torch.as_tensor(center),
            truncation_distance=config.truncation_distance,
            device=config.device,
            grid_shape=config.native_grid_shape,
            enable_dynamic=True,
            enable_static=config.enable_static,
            static_obstacle_color=tuple(float(v) / 255.0 if float(v) > 1 else float(v)
                                        for v in config.static_obstacle_color),
            block_size=config.block_size,
            feature_dim=0,
            color_grid_size=config.color_grid_size,
            accumulator_w_max=config.accumulator_w_max,
        ), self._mapper)
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)
        # The bounded dense mapper updates its exact ESDF together with every
        # map mutation.  Keep the high-level cache/lifecycle contract separate
        # from that storage so callers can reliably tell whether they need to
        # re-request the collision VoxelGrid after integration, clearing, or
        # static-scene replacement.
        self._last_voxel_grid: Optional[VoxelGrid] = None
        self._last_esdf_generation: Optional[torch.Tensor] = None
        self._frame_count = 0
        self._esdf_compute_count = 0

    @property
    def _portable_device(self): return self._mapper.state.tsdf.device

    @property
    def _portable_voxel_size(self): return self.config.voxel_size

    @property
    def tsdf(self) -> BlockSparseTSDF:
        """Dense-backed :class:`BlockSparseTSDF` lifecycle facade.

        The returned object is intentionally not a Warp hash table.  Its
        ``state`` property is the actual dense PyTorch map on CPU or MPS.
        """
        return self._storage

    @property
    def integrator(self) -> BlockSparseESDFIntegrator:
        """Advanced dense mapper facade.

        The portable implementation has no independent Warp block-pool
        integrator object: this mapper owns the actual CPU/MPS state.  It
        nevertheless exposes the source-facing ``integrator`` property so
        callers can use mapping lifecycle operations through a stable object.
        """
        return self

    @property
    def _portable_is_esdf_current(self) -> bool:
        """Whether :meth:`compute_esdf` has observed the current map state."""
        return self._last_esdf_generation is not None and bool(torch.equal(
            self._last_esdf_generation, self._mapper.state.generation,
        ))

    def _normalize_integrate_observations(self, args: tuple, *, observation: Optional[Union[CameraObservation, LidarObservation]], camera_observation: Optional[CameraObservation], lidar_observation: Optional[LidarObservation]) -> tuple[Optional[CameraObservation], Optional[LidarObservation]]:
        """Apply the V2 observation-selection contract before touching state.

        Keeping this validation in the high-level mapper is important: an
        invalid alias combination must not accidentally advance map generation
        or invalidate an otherwise usable ESDF cache.
        """
        if len(args) > 1:
            raise TypeError(f"integrate() takes at most one positional observation, got {len(args)}")
        if args:
            if observation is not None or camera_observation is not None or lidar_observation is not None:
                raise TypeError("positional observation cannot be combined with observation=, camera_observation=, or lidar_observation=")
            observation = args[0]
        elif observation is not None and (camera_observation is not None or lidar_observation is not None):
            raise TypeError("observation= cannot be combined with camera_observation= or lidar_observation=")

        if observation is not None:
            if isinstance(observation, CameraObservation):
                camera_observation = observation
            elif isinstance(observation, LidarObservation):
                lidar_observation = observation
            else:
                raise TypeError(
                    "observation must be CameraObservation or LidarObservation, got "
                    f"{type(observation).__name__}"
                )
        if camera_observation is None and lidar_observation is None:
            raise TypeError("integrate() requires a camera_observation, lidar_observation, or one positional observation")
        return camera_observation, lidar_observation

    def _invalidate_esdf_cache(self):
        self._last_voxel_grid = None
        self._last_esdf_generation = None

    def _strip_static_layer(self, *, rebuild_esdf: bool) -> None:
        """Recover dynamic map state below the dense static overlay."""
        if not bool(self._static_mask.any().item()):
            return
        state = self._mapper.state
        tsdf = torch.where(self._static_mask, torch.ones_like(state.tsdf), state.tsdf)
        weight = torch.where(self._static_mask, torch.zeros_like(state.weight), state.weight)
        occupancy = state.occupancy & ~self._static_mask
        if rebuild_esdf:
            esdf, gradient = dense_esdf(
                occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
                dtype=state.tsdf.dtype,
            )
        else:
            esdf, gradient = state.esdf, state.gradient
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy,
                            esdf=esdf, gradient=gradient, generation=state.generation)

    def _apply_static_layer(self) -> None:
        """Compose the portable static channel over current dynamic state."""
        if not bool(self._static_mask.any().item()):
            return
        state = self._mapper.state
        tsdf = torch.where(self._static_mask, torch.full_like(state.tsdf, -0.5), state.tsdf)
        weight = torch.where(self._static_mask, torch.ones_like(state.weight), state.weight)
        occupancy = state.occupancy | self._static_mask
        esdf, gradient = dense_esdf(
            occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
            dtype=state.tsdf.dtype,
        )
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy,
                            esdf=esdf, gradient=gradient, generation=state.generation)

    def integrate(self, *args, observation: Optional[Union[CameraObservation, LidarObservation]] = None, camera_observation: Optional[CameraObservation] = None, lidar_observation: Optional[LidarObservation] = None) -> None:
        camera_observation, lidar_observation = self._normalize_integrate_observations(
            args, observation=observation, camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        if lidar_observation is not None:
            raise NotImplementedError("portable Mapper LiDAR integration requires CUDA/Warp")
        obs = camera_observation
        if obs.depth_image is None or obs.intrinsics is None or obs.pose is None:
            raise ValueError("camera observation requires depth_image, intrinsics, and pose")
        depth = obs.depth_image.to(self.device, dtype=self._mapper.state.tsdf.dtype) * obs.depth_to_meter
        intrinsics = obs.intrinsics.to(self.device, dtype=depth.dtype)
        pose = obs.pose.get_matrix().to(self.device, dtype=depth.dtype)
        # The upstream mapper carries static geometry in an independent
        # channel.  Remove our dense overlay before depth fusion so its fixed
        # TSDF weight is never mistaken for a camera measurement.
        has_static = bool(self._static_mask.any().item())
        if has_static:
            self._strip_static_layer(rebuild_esdf=False)
        self._mapper.update(NativeObservation(depth, intrinsics, pose))
        if has_static:
            self._apply_static_layer()
        self._frame_count += 1
        self._invalidate_esdf_cache()

    def compute_esdf(self, esdf_origin: Optional[torch.Tensor] = None, esdf_voxel_size: Optional[float] = None) -> VoxelGrid:
        if esdf_origin is None and esdf_voxel_size is None and self.is_esdf_current:
            return self._last_voxel_grid
        if esdf_voxel_size is not None and esdf_voxel_size != self.config.voxel_size:
            raise NotImplementedError(
                "portable dense Mapper computes ESDF at voxel_size; resampling is unavailable"
            )
        if esdf_origin is not None:
            expected = torch.as_tensor(self._mapper.config.grid_center, device=self.device,
                                       dtype=self._mapper.state.tsdf.dtype)
            supplied = torch.as_tensor(esdf_origin, device=self.device, dtype=expected.dtype)
            if supplied.shape != (3,):
                raise ValueError("esdf_origin must be an xyz vector")
            if not torch.allclose(supplied, expected):
                raise NotImplementedError("portable dense Mapper does not implement sliding ESDF windows")
        self._last_voxel_grid = VoxelGrid(
            name="mapper_esdf", pose=[*self._mapper.config.grid_center,1,0,0,0],
            dims=list(self.config.get_actual_extent()),
            voxel_size=self.config.voxel_size,
            feature_tensor=self._mapper.state.esdf[0],
        )
        self._last_esdf_generation = self._mapper.state.generation.clone()
        self._esdf_compute_count += 1
        return self._last_voxel_grid

    def _portable_get_voxel_grid(self):
        """Return a current collision grid, refreshing the derived cache if needed."""
        if not self.is_esdf_current:
            return self.compute_esdf()
        return self._last_voxel_grid

    def _portable_query(self, points: torch.Tensor, *, env_indices: Optional[torch.Tensor] = None,
              padding: float = 0.0):
        """Differentiably query the current dense ESDF on the mapper device."""
        if not self.is_esdf_current:
            self.compute_esdf()
        return self._mapper.query(points, env_indices=env_indices, padding=padding)

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = True) -> Mesh:
        result = self._mapper.extract_mesh()
        return Mesh(
            name="mapper_mesh", pose=[0,0,0,1,0,0,0],
            vertices=result.vertices, faces=result.faces,
        )

    def extract_occupied_voxels(self, surface_only: bool = True, sdf_threshold: Optional[float] = None, *,
                                subvoxel_factor: int = 1, max_points: Optional[int] = None,
                                texture_observations: CameraObservation | Sequence[CameraObservation] | None = None, camera_min_distance: Optional[float] = None,
                                camera_max_distance: Optional[float] = None, texture_depth_tolerance_m: Optional[float] = None) -> OccupiedVoxels:
        integrator = getattr(self, "_integrator", None)
        if integrator is not None and integrator is not self:
            return integrator.extract_occupied_voxels(
                surface_only=surface_only,
                sdf_threshold=sdf_threshold,
                subvoxel_factor=subvoxel_factor,
                max_points=max_points,
                texture_observations=texture_observations,
                camera_min_distance=camera_min_distance,
                camera_max_distance=camera_max_distance,
                texture_depth_tolerance_m=texture_depth_tolerance_m,
            )
        del texture_observations, camera_min_distance, camera_max_distance, texture_depth_tolerance_m
        if subvoxel_factor < 1:
            raise ValueError("subvoxel_factor must be positive")
        state = self._mapper.state
        occupied = state.occupancy[0]
        if surface_only:
            threshold = self.config.truncation_distance if sdf_threshold is None else float(sdf_threshold)
            occupied = occupied & (state.tsdf[0].abs() <= threshold / self.config.truncation_distance)
        coordinates = torch.nonzero(occupied, as_tuple=False)
        centers = coordinates.to(self._mapper.state.tsdf.dtype)
        centers = (centers-(centers.new_tensor(occupied.shape)-1)/2)*self.config.voxel_size
        centers += centers.new_tensor(self._mapper.config.grid_center)
        # The source contract uses evenly spaced subvoxel centers.  Cap source
        # voxels before expansion so every retained source voxel remains
        # complete and the result is deterministic.
        source_limit = None
        if max_points is not None:
            source_limit = max_points // (subvoxel_factor ** 3)
            coordinates, centers = coordinates[:source_limit], centers[:source_limit]
        if subvoxel_factor > 1 and len(centers):
            axis = ((torch.arange(subvoxel_factor, device=centers.device, dtype=centers.dtype) + 0.5)
                    / subvoxel_factor - 0.5) * self.config.voxel_size
            offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            centers = (centers[:, None] + offsets[None]).reshape(-1, 3)
            coordinates = coordinates[:, None].expand(-1, len(offsets), -1).reshape(-1, 3)
        shape = tuple(int(v) for v in occupied.shape)
        indices = (coordinates[:, 0] * shape[1] * shape[2] + coordinates[:, 1] * shape[2] + coordinates[:, 2]).to(torch.long)
        total = int(math.prod(shape))
        coords = torch.stack(torch.meshgrid(
            *[torch.arange(v, device=centers.device, dtype=torch.int32) for v in shape], indexing="ij",
        ), -1).reshape(-1, 3)
        rgb = torch.zeros((total, 1, 4), device=centers.device, dtype=centers.dtype)
        features = centers.new_empty((total, 1, 0))
        feature_weight = centers.new_zeros((total, 1))
        view = BlockDataView(rgb, coords, total, centers.new_tensor(self._mapper.config.grid_center),
                             self.config.voxel_size, 1, shape, features=features,
                             feature_weight=feature_weight)
        return OccupiedVoxels(centers, indices, view, subvoxel_factor=subvoxel_factor)

    def reset(self) -> None:
        self._storage.reset()
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)
        self._frame_count = 0
        self._esdf_compute_count = 0
        self._invalidate_esdf_cache()

    def save_blocks(self, file_path: Union[str, PathLike[str]]) -> None:
        """Persist dense CPU/MPS state in the cuRobo checkpoint envelope."""
        blocks = {
            field: getattr(self._mapper.state, field)
            for field in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")
        }
        save_block_checkpoint(file_path, build_block_metadata(self.tsdf), blocks)

    def import_blocks(self, file_path: Union[str, PathLike[str]], import_weight: Optional[float] = None) -> int:
        """Restore a dense checkpoint into an empty mapper.

        A V2 CUDA/Warp sparse block-pool payload is identified and validated
        by the checkpoint helpers, then rejected explicitly rather than being
        misinterpreted as portable dense state.
        """
        checkpoint = load_block_checkpoint(file_path)
        validate_block_metadata_for_target(checkpoint["block_metadata"], self.tsdf)
        blocks = prepare_blocks_for_import(
            checkpoint["blocks"], checkpoint["block_metadata"],
            import_weight=import_weight,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            block_empty_threshold=0.0,
        )
        if not is_portable_dense_block_payload(blocks):
            raise NotImplementedError("importing CUDA/Warp sparse block-pool payloads is unavailable on the portable dense mapper")
        if bool((self._mapper.state.weight > 0).any().item()):
            raise ValueError("block import requires an empty target mapper")
        state_checkpoint = self._mapper.state_dict()
        state_checkpoint.update(blocks)
        self._storage.import_blocks(state_checkpoint)
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)
        self._invalidate_esdf_cache()
        return int((self._mapper.state.weight > 0).sum().item())

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False) -> dict:
        stats = self._storage.get_stats(scan_pool, scan_hash)
        stats.update({
            "occupied_voxels": int(self._mapper.state.occupancy.sum()),
            "static_voxels": int(self._static_mask.sum()),
            "generation": int(self._mapper.state.generation.max()),
            "frame_count": self._frame_count,
            "esdf_compute_count": self._esdf_compute_count,
            "esdf_current": self.is_esdf_current,
            "esdf_storage": "dense_exact",
        })
        return stats

    def memory_usage_mb(self) -> float:
        return self._storage.memory_usage_mb()

    def _canonical_render_inputs(self, intrinsics: torch.Tensor, pose):
        """Normalize source-supported camera batches without host copies."""
        intrinsics = torch.as_tensor(intrinsics, device=self.device, dtype=self._mapper.state.tsdf.dtype)
        intrinsics_was_batched = intrinsics.ndim == 3 or (intrinsics.ndim == 2 and intrinsics.shape[-1] == 4)
        if intrinsics.ndim == 1:
            if intrinsics.shape != (4,):
                raise ValueError("intrinsics vector must be [fx, fy, cx, cy]")
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.ndim == 2 and intrinsics.shape[-2:] == (3, 3):
            intrinsics = intrinsics.unsqueeze(0)
        elif intrinsics.ndim == 2 and intrinsics.shape[-1] == 4:
            fx, fy, cx, cy = intrinsics.unbind(-1)
            zero = torch.zeros_like(fx)
            one = torch.ones_like(fx)
            intrinsics = torch.stack((
                torch.stack((fx, zero, cx), -1),
                torch.stack((zero, fy, cy), -1),
                torch.stack((zero, zero, one), -1),
            ), -2)
        elif intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must have shape [3,3], [4], [N,3,3], or [N,4]")

        matrix = pose.get_matrix() if hasattr(pose, "get_matrix") else torch.as_tensor(pose)
        matrix = matrix.to(self.device, dtype=intrinsics.dtype)
        # ``Pose.from_list`` stores one transform as ``[1,4,4]`` internally;
        # retain the established Mapper convention that this still renders an
        # unbatched image.  Actual multi-camera pose batches expose B > 1.
        pose_was_batched = matrix.ndim == 3 and matrix.shape[0] > 1
        if matrix.ndim == 2:
            if matrix.shape != (4, 4):
                raise ValueError("pose matrix must have shape [4,4] or [N,4,4]")
            matrix = matrix.unsqueeze(0)
        elif matrix.ndim != 3 or matrix.shape[-2:] != (4, 4):
            raise ValueError("pose matrix must have shape [4,4] or [N,4,4]")
        count = max(intrinsics.shape[0], matrix.shape[0])
        if intrinsics.shape[0] not in (1, count) or matrix.shape[0] not in (1, count):
            raise ValueError("intrinsics and pose batch sizes must match or be one")
        if intrinsics.shape[0] == 1 and count > 1:
            intrinsics = intrinsics.expand(count, -1, -1)
        if matrix.shape[0] == 1 and count > 1:
            matrix = matrix.expand(count, -1, -1)
        return intrinsics, matrix, intrinsics_was_batched or pose_was_batched

    def _render_result(self, intrinsics: torch.Tensor, pose, image_shape):
        intrinsics, matrix, batched = self._canonical_render_inputs(intrinsics, pose)
        if batched:
            raise ValueError("_render_result is an unbatched compatibility helper; use render() for camera batches")
        return self._mapper.render(intrinsics[0], matrix[0], image_shape)

    def render_depth(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render(intrinsics, pose, image_shape)[0]

    def render(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsics, matrices, batched = self._canonical_render_inputs(intrinsics, pose)
        results = [self._mapper.render(intrinsics[index], matrices[index], image_shape)
                   for index in range(intrinsics.shape[0])]
        depth = torch.stack([result.depth for result in results])
        valid = torch.stack([result.valid for result in results])
        # Exact dense ESDF gradients are the available portable normal field;
        # pixels without an observed hit intentionally retain a zero normal.
        normals = torch.zeros(depth.shape + (3,), device=depth.device, dtype=depth.dtype)
        if not batched:
            return depth[0], normals[0], valid[0]
        return depth, normals, valid

    def render_color(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        # Feature/color fusion is explicitly unavailable, therefore the
        # untextured color channel is deterministic black rather than invented.
        color = torch.zeros(depth.shape + (3,), device=depth.device, dtype=torch.uint8)
        return depth, normals, color, valid

    def render_color_only(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render_color(intrinsics, pose, image_shape)[2]

    def render_depth_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(
            depth, self.config.depth_minimum_distance, self.config.depth_maximum_distance, valid,
        )

    def render_normal_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)

    def render_shaded(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int],
                      light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0), ambient: float = 1.0,
                      use_color: bool = True) -> torch.Tensor:
        del use_color
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        light = normals.new_tensor(light_direction)
        light = light / torch.linalg.vector_norm(light).clamp_min(torch.finfo(normals.dtype).eps)
        intensity = (normals * light).sum(-1).clamp_min(0) * (1 - ambient) + ambient
        return torch.where(valid[..., None], (intensity[..., None] * 255).to(torch.uint8),
                           torch.zeros_like(normals, dtype=torch.uint8))

    def extract_textured_mesh(self, texture_observations: CameraObservation | Sequence[CameraObservation], refine_iterations: int = 0,
                              surface_only: bool = True, camera_min_distance: Optional[float] = None,
                              camera_max_distance: Optional[float] = None, texture_depth_tolerance_m: Optional[float] = None) -> Mesh:
        integrator = getattr(self, "_integrator", None)
        if integrator is not None and integrator is not self:
            return integrator.extract_textured_mesh(
                texture_observations=texture_observations,
                refine_iterations=refine_iterations,
                surface_only=surface_only,
                camera_min_distance=camera_min_distance,
                camera_max_distance=camera_max_distance,
                texture_depth_tolerance_m=texture_depth_tolerance_m,
            )
        del texture_observations, camera_min_distance, camera_max_distance, texture_depth_tolerance_m
        # Mesh geometry remains valid even when no texture feature volume was
        # requested.  The returned Mesh contains no invented vertex colors.
        return self.extract_mesh(refine_iterations, surface_only)

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> MatchedVoxels:
        del feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector
        raise NotImplementedError("feature-volume matching requires an enabled portable feature mapper")

    def get_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                    surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                    minimum_score: Optional[float] = None,
                                    feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> MatchedVoxels:
        return self.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector,
        )

    def _replace_state(self, **fields):
        state = self._mapper.state
        values = {name: fields.get(name, getattr(state, name)) for name in
                  ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")}
        self._mapper.state = DenseMap(**values)

    def clear_region(self, bounds_min, bounds_max) -> int:
        lower = torch.as_tensor(bounds_min, device=self.device, dtype=self._mapper.state.tsdf.dtype)
        upper = torch.as_tensor(bounds_max, device=self.device, dtype=self._mapper.state.tsdf.dtype)
        if lower.shape != (3,) or upper.shape != (3,) or bool((upper < lower).any().item()):
            raise ValueError("bounds must be xyz vectors with bounds_max >= bounds_min")
        origin = self.config.origin.to(device=self.device, dtype=lower.dtype)
        shape = torch.tensor(self.config.native_grid_shape, device=self.device)
        # Native storage is x,y,z while this compatibility config exposes
        # world xyz; both are deliberately kept in the same order here.
        start = torch.floor((lower - origin) / self.config.voxel_size).to(torch.long).clamp_min(0)
        stop = (torch.floor((upper - origin) / self.config.voxel_size).to(torch.long) + 1).minimum(shape)
        slices = tuple(slice(int(start[index]), int(stop[index])) for index in range(3))
        state = self._mapper.state
        mask = torch.zeros_like(state.occupancy)
        mask[(slice(None),) + slices] = True
        # Source clear operations target the dynamic channel.  Static scene
        # occupancy survives until update_static_obstacles() or reset().
        mask &= ~self._static_mask
        count = int((state.weight[mask] > 0).sum().item())
        tsdf = torch.where(mask, torch.ones_like(state.tsdf), state.tsdf)
        weight = torch.where(mask, torch.zeros_like(state.weight), state.weight)
        occupancy = state.occupancy & ~mask
        esdf, gradient = dense_esdf(occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
                                    dtype=state.tsdf.dtype)
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy, esdf=esdf,
                            gradient=gradient, generation=state.generation + (mask.any()).to(torch.int64))
        if count:
            self._invalidate_esdf_cache()
        return count

    def clear_blocks(self, pool_indices) -> int:
        index = torch.as_tensor(pool_indices, device=self.device, dtype=torch.long).reshape(-1)
        size = int(torch.tensor(self.config.native_grid_shape).prod().item())
        if bool(((index < 0) | (index >= size)).any().item()):
            raise ValueError("portable dense block indices must index the flattened grid")
        coordinates = torch.stack(torch.unravel_index(index, self.config.native_grid_shape), -1)
        count = 0
        for coordinate in coordinates.tolist():
            lower = self.config.origin + torch.tensor(coordinate) * self.config.voxel_size
            count += self.clear_region(lower, lower)
        return count

    @classmethod
    def load_blocks(cls, file_path: Union[str, PathLike[str]], target_cfg: MapperCfg, import_weight: Optional[float] = None) -> "Mapper":
        result = cls(target_cfg)
        result.import_blocks(file_path, import_weight)
        return result

    def update_static_obstacles(self, scene: SceneData, env_idx: int = 0) -> None:
        """Stamp portable cuboids and spheres into the dense TSDF occupancy.

        This is deliberately bounded to analytic primitives.  CUDA/Warp scene
        tensors, mesh BVHs, and sparse static channels have materially
        different memory and tie semantics, so they reject explicitly.
        """
        if not self.config.enable_static:
            raise RuntimeError("MapperCfg.enable_static=True is required before stamping static obstacles")
        from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
        from curobo._src.geom.data.data_scene import SceneData

        if isinstance(scene, SceneData):
            scene = scene.scene_model
        if isinstance(scene, (list, tuple)):
            if env_idx < 0 or env_idx >= len(scene):
                raise ValueError("env_idx is outside the provided static scene list")
            scene = scene[env_idx]
        if not isinstance(scene, SceneCfg):
            raise TypeError("static stamping requires SceneCfg or SceneData carrying SceneCfg")
        if scene.mesh or scene.voxel or scene.capsule or scene.cylinder:
            raise NotImplementedError("portable static stamping supports only cuboids and spheres; mesh/voxel/Warp primitives require CUDA/Warp")
        state = self._mapper.state
        centers = torch.stack(torch.meshgrid(
            *[(torch.arange(n, device=self.device, dtype=state.tsdf.dtype) - (n - 1) / 2) * self.config.voxel_size + c
              for n, c in zip(self.config.native_grid_shape, self._mapper.config.grid_center)], indexing="ij",
        ), -1)
        mask = torch.zeros_like(state.occupancy[0])
        for obstacle in scene.cuboid:
            if not isinstance(obstacle, Cuboid):
                raise TypeError("scene.cuboid must contain Cuboid records")
            pose = obstacle.pose
            position = centers.new_tensor(pose[:3])
            quat = centers.new_tensor(pose[3:])
            quat = quat / torch.linalg.vector_norm(quat).clamp_min(torch.finfo(centers.dtype).eps)
            w, x, y, z = quat.unbind()
            rotation = torch.stack((
                1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
            )).reshape(3, 3)
            local = (centers - position) @ rotation
            half = centers.new_tensor(obstacle.dims) / 2
            mask |= (local.abs() <= half).all(-1)
        for obstacle in scene.sphere:
            if not isinstance(obstacle, Sphere):
                raise TypeError("scene.sphere must contain Sphere records")
            position = centers.new_tensor(obstacle.position if obstacle.position is not None else obstacle.pose[:3])
            mask |= torch.linalg.vector_norm(centers - position, dim=-1) <= obstacle.radius
        # Remove the previous overlay before replacing it.  The helpers keep
        # the underlying dynamic weights separate across camera updates.
        self._strip_static_layer(rebuild_esdf=False)
        state = self._mapper.state
        occupancy = state.occupancy.clone()
        occupancy[0] |= mask
        tsdf = state.tsdf.clone()
        weight = state.weight.clone()
        self._static_mask = torch.zeros_like(occupancy)
        self._static_mask[0] = mask
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy,
                            esdf=state.esdf, gradient=state.gradient,
                            generation=state.generation + 1)
        self._apply_static_layer()
        self._invalidate_esdf_cache()
        return int(mask.sum().item())


# Dense-only convenience surface retained at runtime without adding public
# methods to the pinned Mapper declaration contract.
Mapper.device = Mapper._portable_device
Mapper.voxel_size = Mapper._portable_voxel_size
Mapper.is_esdf_current = Mapper._portable_is_esdf_current
Mapper.get_voxel_grid = Mapper._portable_get_voxel_grid
Mapper.query = Mapper._portable_query
