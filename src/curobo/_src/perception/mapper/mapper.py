from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Tuple, Union
import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.constants import resolve_feature_integration_kernel
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
    is_sparse_block_payload,
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


# The portable high-level mapper keeps a differentiable dense TSDF/ESDF
# mirror for collision queries.  Its storage therefore scales with the voxel
# volume, unlike the bounded sparse pool used for RGB/features.  Fail before
# allocating that mirror when a configuration would require an unreasonable
# amount of resident memory.  This is deliberately a preflight guard (rather
# than catching an allocator OOM after partial construction), and leaves the
# standalone sparse integrator available for genuinely large maps.
_MAX_DENSE_MAPPER_BYTES = 1 << 30
_DENSE_MAPPER_BYTES_PER_VOXEL = 26  # tsdf, weight, occupancy, esdf, gradient, static mask


def _estimated_dense_mapper_bytes(config: MapperCfg) -> int:
    """Estimate the resident dense mirror required by :class:`Mapper`.

    The estimate is intentionally conservative: ``PerceptionMapper`` stores
    five dense fields (including a 3-vector gradient), while the high-level
    facade owns a static mask.  Temporary tensors used by ESDF rebuilds are
    not included, so this remains a lower bound for peak memory.
    """
    return int(math.prod(config.native_grid_shape) * _DENSE_MAPPER_BYTES_PER_VOXEL)


def _validate_dense_mapper_budget(config: MapperCfg) -> None:
    estimate = _estimated_dense_mapper_bytes(config)
    if estimate > _MAX_DENSE_MAPPER_BYTES:
        mib = estimate / (1024 * 1024)
        raise MemoryError(
            "portable high-level Mapper dense mirror requires approximately "
            f"{mib:.0f} MiB for {tuple(config.native_grid_shape)} voxels, "
            f"exceeding the {_MAX_DENSE_MAPPER_BYTES // (1024 * 1024)} MiB "
            "safety limit; use the bounded sparse integrator or a larger "
            "voxel size/partitioned map"
        )


def _device(value):
    if str(value).startswith("cuda"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return value


class Mapper:
    def __init__(self, config: MapperCfg):
        self.config = config
        _validate_dense_mapper_budget(config)
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
        # The storage facade is backed by the native dense map and is only used
        # for lifecycle metadata here.  Give it a dense-capacity marker so its
        # optional compatibility sparse mirror is not allocated and then
        # immediately discarded below (notably important for large maps).
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
            feature_dim=config.feature_dim,
            feature_block_grid_size=config.feature_block_grid_size,
            feature_grid_height=config.feature_grid_height,
            feature_grid_width=config.feature_grid_width,
            feature_channels_per_thread=config.feature_channels_per_thread,
            color_grid_size=config.color_grid_size,
            accumulator_w_max=config.accumulator_w_max,
        ), self._mapper)
        # Keep the source-visible RGB/feature block pool compact while the
        # native dense field supplies differentiable TSDF/ESDF queries.  The
        # pool is bounded by MapperCfg.max_blocks and therefore does not
        # materialize the nominal voxel volume.
        from curobo._src.perception.mapper.sparse_runtime import PortableSparseTSDF
        sparse_config = SimpleNamespace(
            **config.__dict__,
            # Standalone sparse runtimes use a center anchor and signed block
            # coordinates.  MapperCfg.origin remains the dense lower corner.
            origin=config.grid_center,
            grid_shape=config.grid_shape,
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
        )
        self._portable_sparse = PortableSparseTSDF(sparse_config)
        self._storage._sparse_data = self._portable_sparse.data
        # Appearance is accumulated directly in the sparse block tensors.  Do
        # not allocate four max_blocks-sized shadow arrays here: for nominal
        # 512^3 maps max_blocks is a capacity, not the number of live blocks.
        self._appearance_rgb = self._portable_sparse.data.block_grid_rgb[:0, :1, :3]
        self._appearance_rgb_weight = self._portable_sparse.data.block_grid_rgb[:0, :1, 3]
        self._appearance_features = self._portable_sparse.data.block_features[:0, :1] if self._portable_sparse.data.has_features else None
        self._appearance_feature_weight = self._portable_sparse.data.block_feature_weight[:0, :1] if self._portable_sparse.data.has_features else None
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
        use_tiled_feature_kernel = resolve_feature_integration_kernel(
            config.feature_integration_kernel,
            config.feature_dim,
            config.max_support_pixels_per_block_camera,
        )
        self._last_integration = {
            "num_visible_blocks": 0,
            "support_overflow_count": 0,
            "profile_kernel_timings": bool(config.profile_integration_kernel_timings),
            "use_tiled_feature_kernel": use_tiled_feature_kernel,
        }
        self._last_integration_kernel_timings_ms: dict[str, float] = {}
        visible_capacity = config.max_visible_blocks_per_integration or config.max_blocks
        camera_integrator = SimpleNamespace(
            max_visible_blocks_per_integration=visible_capacity,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            use_tiled_feature_kernel=use_tiled_feature_kernel,
            profile_kernel_timings=config.profile_integration_kernel_timings,
            visible_count=torch.zeros(1, device=self.device, dtype=torch.int32),
            pool_indices=torch.zeros(visible_capacity, device=self.device, dtype=torch.int32),
            support_counts=torch.zeros(
                (visible_capacity, config.num_cameras), device=self.device, dtype=torch.int32
            ),
            support_pixels=torch.zeros(
                (
                    visible_capacity,
                    config.num_cameras,
                    config.max_support_pixels_per_block_camera,
                ),
                device=self.device,
                dtype=torch.int32,
            ),
            clear_pool_indices=torch.zeros(
                config.max_blocks, device=self.device, dtype=torch.int32
            ),
            visible_epoch=torch.zeros(
                config.max_blocks, device=self.device, dtype=torch.int32
            ),
            pool_to_visible_slot=torch.full(
                (config.max_blocks,), -1, device=self.device, dtype=torch.int32
            ),
        )
        self._tsdf_integrator = SimpleNamespace(
            config=config,
            _camera_integrator=camera_integrator,
        )

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
        # Mapper observations use metric depth tensors.  ``depth_to_meter`` is
        # consumed by CameraObservation point-cloud helpers for integer sensor
        # payloads, but the upstream mapper integration ABI receives the
        # already-converted floating depth image directly.
        depth = obs.depth_image.to(self.device, dtype=self._mapper.state.tsdf.dtype)
        intrinsics = obs.intrinsics.to(self.device, dtype=depth.dtype)
        pose = obs.pose.get_matrix().to(self.device, dtype=depth.dtype)
        # The upstream mapper carries static geometry in an independent
        # channel.  Remove our dense overlay before depth fusion so its fixed
        # TSDF weight is never mistaken for a camera measurement.
        has_static = bool(self._static_mask.any().item())
        if has_static:
            self._strip_static_layer(rebuild_esdf=False)
        self._mapper.update(NativeObservation(depth, intrinsics, pose))
        if self.config.decay_factor < 1.0:
            self._portable_sparse.decay_and_recycle(self.config.decay_factor)
        # Geometry allocation/fusion and appearance support use separate
        # paths.  The sparse runtime's compact fallback can fuse an RGB image
        # itself, but doing so here would count every pixel once more in the
        # support-aware accumulator below.
        sparse_obs = CameraObservation(
            depth_image=depth,
            intrinsics=intrinsics,
            pose=obs.pose,
        )
        self._portable_sparse.integrate(sparse_obs)
        self._update_portable_camera_support(obs)
        if has_static:
            self._apply_static_layer()
        self._frame_count += 1
        valid_pixels = torch.isfinite(depth) & (depth >= self.config.depth_minimum_distance) & (depth <= self.config.depth_maximum_distance)
        visible_blocks = self._storage._logical_block_count()[0] if bool(valid_pixels.any().item()) else 0
        support_overflow = 0
        if visible_blocks:
            # A bounded support list overflows whenever a projected block has
            # more contributing pixels than the configured per-camera cap.
            support_overflow = max(
                0,
                int(valid_pixels.sum().item())
                - visible_blocks * self.config.max_support_pixels_per_block_camera,
            )
        self._last_integration = {
            "num_visible_blocks": visible_blocks,
            "support_overflow_count": support_overflow,
            "profile_kernel_timings": bool(self.config.profile_integration_kernel_timings),
            "use_tiled_feature_kernel": self._tsdf_integrator._camera_integrator.use_tiled_feature_kernel,
        }
        self._last_integration_kernel_timings_ms = (
            {"portable_depth_fusion": 0.0}
            if self.config.profile_integration_kernel_timings
            else {}
        )
        self._invalidate_esdf_cache()

    def _update_portable_camera_support(self, observation: CameraObservation) -> None:
        """Expose deterministic per-block camera support diagnostics.

        These tensors are observable in upstream correctness tooling and are
        also useful for feature/RGB auditing.  The portable pool records one
        representative projected pixel per block/camera; accumulation itself
        remains independent of this diagnostic view.
        """
        data = self._portable_sparse.data
        # ``num_allocated`` is a high-water mark; cleared blocks remain holes.
        # Diagnostics and appearance updates must operate on the active pool
        # slots, otherwise a clear followed by a new frame can overwrite the
        # wrong appearance entry.
        active_pools = sorted(self._portable_sparse._pool_to_coord)
        count = len(active_pools)
        cameras = self.config.num_cameras
        device = data.block_data.device
        camera_integrator = self._tsdf_integrator._camera_integrator
        visible_count = min(count, camera_integrator.max_visible_blocks_per_integration)
        pool = torch.as_tensor(active_pools[:visible_count], device=device, dtype=torch.int32)
        counts = torch.zeros((visible_count, cameras), device=device, dtype=torch.int32)
        pixels = torch.zeros((visible_count, cameras, 1), device=device, dtype=torch.int32)
        if visible_count:
            coords = data.block_coords.view(-1, 3)[pool.long()].float()
            grid_d, grid_h, grid_w = (int(value) for value in data.grid_shape)
            block_size = int(data.block_size)
            block_offsets = torch.tensor(
                [math.ceil(grid_w / block_size) // 2,
                 math.ceil(grid_h / block_size) // 2,
                 math.ceil(grid_d / block_size) // 2],
                device=device, dtype=torch.float32,
            )
            center_offset = torch.tensor([grid_w, grid_h, grid_d], device=device,
                                         dtype=torch.float32) * 0.5
            voxel = (coords + block_offsets) * block_size + block_size * 0.5
            world = data.origin.to(device=device, dtype=torch.float32) + (
                voxel - center_offset
            ) * float(data.voxel_size)
            matrices = observation.pose.get_matrix().to(device=device, dtype=torch.float32)
            if matrices.ndim == 2:
                matrices = matrices.unsqueeze(0)
            intrinsics = observation.intrinsics.to(device=device, dtype=torch.float32)
            if intrinsics.ndim == 2:
                intrinsics = intrinsics.unsqueeze(0)
            depth = observation.depth_image.to(device=device, dtype=torch.float32)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            height, width = depth.shape[-2:]
            for camera in range(min(cameras, depth.shape[0])):
                matrix = matrices[min(camera, len(matrices) - 1)]
                local = (world - matrix[:3, 3]) @ matrix[:3, :3]
                z = local[:, 2]
                safe_z = z.clamp_min(torch.finfo(torch.float32).tiny)
                k = intrinsics[min(camera, len(intrinsics) - 1)]
                uf = k[0, 0] * local[:, 0] / safe_z + k[0, 2]
                vf = k[1, 1] * local[:, 1] / safe_z + k[1, 2]
                px = torch.floor(uf + 0.5).long()
                py = torch.floor(vf + 0.5).long()
                inside = (z > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
                linear = py.clamp(0, height - 1) * width + px.clamp(0, width - 1)
                sampled = depth[camera].reshape(-1)[linear]
                supported = inside & torch.isfinite(sampled) & (
                    (sampled - z).abs() <= self.config.truncation_distance
                )
                counts[:, camera] = supported.to(torch.int32)
                pixels[:, camera, 0] = linear.to(torch.int32)
                if observation.rgb_image is not None and bool(supported.any().item()):
                    rgb = observation.rgb_image.to(device=device, dtype=torch.float32)
                    if rgb.ndim == 3:
                        rgb = rgb.unsqueeze(0)
                    image = rgb[camera] / 255.0
                    x0 = torch.floor(uf).long().clamp(0, width - 1)
                    y0 = torch.floor(vf).long().clamp(0, height - 1)
                    x1 = (x0 + 1).clamp(max=width - 1)
                    y1 = (y0 + 1).clamp(max=height - 1)
                    tx = (uf.clamp(0, width - 1) - x0.float()).unsqueeze(-1)
                    ty = (vf.clamp(0, height - 1) - y0.float()).unsqueeze(-1)
                    color = (
                        image[y0, x0] * (1 - tx) * (1 - ty)
                        + image[y0, x1] * tx * (1 - ty)
                        + image[y1, x0] * (1 - tx) * ty
                        + image[y1, x1] * tx * ty
                    )
                    selected = torch.nonzero(supported, as_tuple=False).flatten()
                    rgb_sum = data.block_grid_rgb[:, 0, :3]
                    rgb_weight = data.block_grid_rgb[:, 0, 3]
                    rgb_sum.index_add_(0, pool.long()[selected], color[selected].to(rgb_sum.dtype))
                    rgb_weight.index_add_(0, pool.long()[selected], torch.ones_like(selected, dtype=rgb_weight.dtype))
                feature_grid = getattr(observation, "feature_grid", None)
                if data.has_features and feature_grid is not None and bool(supported.any().item()):
                    features = feature_grid.to(device=device, dtype=torch.float32)
                    if features.ndim == 3:
                        features = features.unsqueeze(0)
                    feature_h, feature_w = features.shape[1:3]
                    gy = ((py.clamp(0, height - 1) * feature_h) // height).clamp(0, feature_h - 1)
                    gx = ((px.clamp(0, width - 1) * feature_w) // width).clamp(0, feature_w - 1)
                    values = features[camera, gy, gx]
                    selected = torch.nonzero(supported, as_tuple=False).flatten()
                    feature_sum = data.block_features[:, 0]
                    feature_weight = data.block_feature_weight[:, 0]
                    feature_sum.index_add_(0, pool.long()[selected], values[selected].to(feature_sum.dtype))
                    feature_weight.index_add_(0, pool.long()[selected], torch.ones_like(selected, dtype=feature_weight.dtype))
        camera_integrator.visible_count.fill_(visible_count)
        camera_integrator.pool_indices.zero_()
        camera_integrator.pool_indices[:visible_count].copy_(pool)
        camera_integrator.support_counts.zero_()
        camera_integrator.support_counts[:visible_count].copy_(counts)
        camera_integrator.support_pixels.zero_()
        camera_integrator.support_pixels[:visible_count, :, 0].copy_(pixels[:, :, 0])

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
        if self._portable_sparse._pool_to_coord:
            from curobo._src.perception.mapper.mesh_extractor import extract_mesh_block_sparse
            vertices, faces, normals, colors = extract_mesh_block_sparse(
                self._portable_sparse, surface_only=surface_only,
                refine_iterations=refine_iterations,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
                return_faces=True,
            )
            return Mesh(name="mapper_mesh", pose=[0, 0, 0, 1, 0, 0, 0],
                        vertices=vertices, faces=faces,
                        vertex_normals=normals, vertex_colors=colors)
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
        # The sparse appearance pool is authoritative for source-shaped
        # extraction.  Dense geometry remains useful for collision queries,
        # but returning a fabricated all-zero BlockDataView loses RGB/features.
        if texture_observations is None:
            return self._extract_sparse_occupied_voxels(
                surface_only=surface_only, sdf_threshold=sdf_threshold,
                subvoxel_factor=subvoxel_factor, max_points=max_points,
            )
        del camera_min_distance, camera_max_distance, texture_depth_tolerance_m
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

    def _extract_sparse_occupied_voxels(self, *, surface_only: bool,
                                        sdf_threshold: Optional[float],
                                        subvoxel_factor: int,
                                        max_points: Optional[int]) -> OccupiedVoxels:
        data = self._portable_sparse.data
        pools = sorted(self._portable_sparse._pool_to_coord)
        view = BlockDataView(
            rgb_grid=data.block_grid_rgb, coords=data.block_coords,
            num_allocated=data.block_data.shape[0], origin=data.origin,
            voxel_size=float(data.voxel_size), block_size=int(data.block_size),
            grid_shape=tuple(data.grid_shape), color_grid_size=int(data.color_grid_size),
            feature_block_grid_size=int(data.feature_block_grid_size),
            features=data.block_features, feature_weight=data.block_feature_weight,
            feature_dim=int(data.feature_dim),
        )
        if not pools:
            empty = torch.empty((0, 3), device=self.device, dtype=self._mapper.state.tsdf.dtype)
            return OccupiedVoxels(empty, torch.empty((0,), device=self.device, dtype=torch.int32), view,
                                  subvoxel_factor=subvoxel_factor)
        pool = torch.as_tensor(pools, device=self.device, dtype=torch.long)
        values = data.block_data[pool].float()
        weights = values[..., 1]
        sdf = values[..., 0] / weights.clamp_min(1e-6)
        valid = weights >= float(self.config.minimum_tsdf_weight)
        if data.has_static:
            static = data.static_block_data[pool].float()
            static_valid = torch.isfinite(static) & (static < 1e9)
            sdf = torch.minimum(sdf, static)
            valid |= static_valid
        threshold = self.config.truncation_distance if sdf_threshold is None else float(sdf_threshold)
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("sdf_threshold must be finite and non-negative")
        # A +1 TSDF denotes saturated free space, not a surface.  Use a
        # strict threshold here (matching the upstream block extractor) so a
        # threshold equal to the truncation distance does not include every
        # free-space voxel in front of the observed surface.
        keep = valid & ((sdf.abs() < threshold / float(data.truncation_distance)) if surface_only else (sdf <= 0))
        block_rank, local_idx = torch.nonzero(keep, as_tuple=True)
        bs = int(data.block_size)
        lx = torch.div(local_idx, bs * bs, rounding_mode="floor")
        rem = local_idx.remainder(bs * bs)
        ly = torch.div(rem, bs, rounding_mode="floor")
        lz = rem.remainder(bs)
        local = torch.stack((lx, ly, lz), -1).to(torch.float32)
        block_coords = data.block_coords.view(-1, 3)[pool[block_rank]].float()
        grid_d, grid_h, grid_w = (int(value) for value in data.grid_shape)
        block_offsets = torch.tensor(
            [math.ceil(grid_w / bs) // 2, math.ceil(grid_h / bs) // 2,
             math.ceil(grid_d / bs) // 2], device=self.device, dtype=torch.float32
        )
        center_offset = torch.tensor([grid_w, grid_h, grid_d], device=self.device,
                                     dtype=torch.float32) * 0.5
        centers = data.origin.to(self.device) + (
            (block_coords + block_offsets) * bs + local + 0.5 - center_offset
        ) * float(data.voxel_size)
        idx = pool[block_rank].to(torch.int32)
        if max_points is not None:
            if max_points < 1:
                raise ValueError("max_points must be a positive integer or None")
            n = min(len(centers), max_points // max(1, subvoxel_factor ** 3))
            centers, idx = centers[:n], idx[:n]
        if subvoxel_factor > 1 and len(centers):
            axis = ((torch.arange(subvoxel_factor, device=self.device, dtype=centers.dtype) + 0.5)
                    / subvoxel_factor - 0.5) * float(self.config.voxel_size)
            offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            centers = (centers[:, None] + offsets[None]).reshape(-1, 3)
            idx = idx.repeat_interleave(len(offsets))
        return OccupiedVoxels(centers, idx, view, subvoxel_factor=subvoxel_factor)

    def reset(self) -> None:
        self._storage.reset()
        self._portable_sparse.reset()
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)
        self._frame_count = 0
        self._esdf_compute_count = 0
        self._last_integration.update(num_visible_blocks=0, support_overflow_count=0)
        self._last_integration_kernel_timings_ms = {}
        self._invalidate_esdf_cache()

    def save_blocks(self, file_path: Union[str, PathLike[str]]) -> None:
        """Persist the compact source-shaped block pool."""
        blocks = self._portable_sparse.export_blocks()
        save_block_checkpoint(file_path, build_block_metadata(self.tsdf), blocks)

    def _sync_dense_from_sparse(self) -> None:
        """Rebuild the bounded dense query field from imported sparse voxels."""
        data = self._portable_sparse.data
        active_pools = sorted(self._portable_sparse._pool_to_coord)
        count = len(active_pools)
        state = self._mapper.state
        tsdf = torch.ones_like(state.tsdf)
        weight = torch.zeros_like(state.weight)
        imported_static = torch.zeros_like(state.occupancy)
        if count:
            block_size = int(data.block_size)
            axis = (torch.arange(block_size, device=self.device, dtype=state.tsdf.dtype) + 0.5)
            local = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            pool_idx = torch.as_tensor(active_pools, device=self.device, dtype=torch.long)
            coords = data.block_coords.view(-1, 3)[pool_idx].to(state.tsdf.dtype)
            grid_d, grid_h, grid_w = (int(value) for value in data.grid_shape)
            block_offsets = coords.new_tensor([
                math.ceil(grid_w / block_size) // 2,
                math.ceil(grid_h / block_size) // 2,
                math.ceil(grid_d / block_size) // 2,
            ])
            center_offset = coords.new_tensor([grid_w, grid_h, grid_d]) * 0.5
            world = data.origin.to(self.device) + (
                (coords[:, None] + block_offsets) * block_size
                + local[None] - center_offset
            ) * data.voxel_size
            shape = world.new_tensor(self._mapper.config.shape)
            indices = torch.floor((world - self.config.origin.to(self.device)) / data.voxel_size).long()
            inside = ((indices >= 0) & (indices < shape.long())).all(-1)
            sums = data.block_data[pool_idx, :, 0].float()
            weights = data.block_data[pool_idx, :, 1].float()
            observed = inside & (weights >= self.config.minimum_tsdf_weight)
            selected = indices[observed]
            selected_weights = weights[observed].to(weight.dtype)
            selected_tsdf = (sums[observed] / weights[observed].clamp_min(1.0e-12)).to(tsdf.dtype)
            tsdf[0, selected[:, 0], selected[:, 1], selected[:, 2]] = selected_tsdf
            weight[0, selected[:, 0], selected[:, 1], selected[:, 2]] = selected_weights
            if data.has_static:
                static = data.static_block_data[pool_idx].float()
                static_valid = torch.isfinite(static) & (static < 1.0e9)
                static_values = static[static_valid]
                static_indices = indices[static_valid]
                inside_static = ((static_indices >= 0) & (static_indices < shape.long())).all(-1)
                static_indices = static_indices[inside_static]
                static_values = static_values[inside_static]
                if static_indices.numel():
                    tsdf[0, static_indices[:, 0], static_indices[:, 1], static_indices[:, 2]] = static_values
                    weight[0, static_indices[:, 0], static_indices[:, 1], static_indices[:, 2]] = 1.0
                    imported_static[0, static_indices[:, 0], static_indices[:, 1], static_indices[:, 2]] = True
        occupancy = (weight > 0) & (tsdf <= self._mapper.config.occupancy_threshold)
        esdf, gradient = dense_esdf(
            occupancy,
            self.config.voxel_size,
            self._mapper.config.unobserved_esdf,
            dtype=tsdf.dtype,
        )
        self._replace_state(
            tsdf=tsdf,
            weight=weight,
            occupancy=occupancy,
            esdf=esdf,
            gradient=gradient,
            generation=state.generation + 1,
        )
        self._static_mask = imported_static

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
        if is_sparse_block_payload(blocks):
            count = self._portable_sparse.import_blocks(blocks)
            data = self._portable_sparse.data
            self._sync_dense_from_sparse()
            self._invalidate_esdf_cache()
            return count
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
            "observed_voxels": int((self._mapper.state.weight > 0).sum()),
            "static_voxels": int(self._static_mask.sum()),
            "generation": int(self._mapper.state.generation.max()),
            "frame_count": self._frame_count,
            "esdf_compute_count": self._esdf_compute_count,
            "esdf_current": self.is_esdf_current,
            "esdf_storage": "dense_exact",
            "last_integration": dict(self._last_integration),
            "last_integration_kernel_timings_ms": dict(self._last_integration_kernel_timings_ms),
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
        # Estimate geometric normals from the rendered camera-space surface.
        # This is deterministic and remains useful for the dense native path,
        # whose depth renderer intentionally returns no normal channel.
        h, w = image_shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij"
        )
        normals = torch.zeros(depth.shape + (3,), device=depth.device, dtype=depth.dtype)
        for index in range(depth.shape[0]):
            d = depth[index]
            k = intrinsics[index]
            px = (xx - k[0, 2]) / k[0, 0]
            py = (yy - k[1, 2]) / k[1, 1]
            points = torch.stack((px * d, py * d, d), -1)
            dx = points[:, 2:] - points[:, :-2]
            dy = points[2:] - points[:-2]
            cross = torch.cross(dx[1:-1], dy[:, 1:-1], dim=-1)
            nrm = cross / torch.linalg.vector_norm(cross, dim=-1, keepdim=True).clamp_min(1e-6)
            normals[index, 1:-1, 1:-1] = nrm
        normals = torch.where(valid[..., None], normals, torch.zeros_like(normals))
        # Sparse voxel z-buffers often have isolated valid pixels, for which a
        # centred finite difference has no two-sided neighbours.  A camera
        # facing normal is the well-defined limit for a locally constant depth
        # patch and keeps the normal channel useful instead of silently zero.
        fallback_normal = torch.zeros_like(normals)
        fallback_normal[..., 2] = 1.0
        normals = torch.where(
            valid[..., None] & (normals.norm(dim=-1, keepdim=True) < 1e-6),
            fallback_normal,
            normals,
        )
        if not batched:
            return depth[0], normals[0], valid[0]
        return depth, normals, valid

    def render_color(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        unbatched = depth.ndim == 2
        depth_b = depth.unsqueeze(0) if unbatched else depth
        color = torch.zeros(depth_b.shape + (3,), device=depth.device, dtype=torch.uint8)
        voxels = self.extract_occupied_voxels(surface_only=False)
        if len(voxels):
            points = voxels.centers
            colors = voxels.colors_uint8().to(device=depth.device)
            _, matrices, _ = self._canonical_render_inputs(intrinsics, pose)
            intr_b, matrices, _ = self._canonical_render_inputs(intrinsics, pose)
            for camera in range(depth_b.shape[0]):
                local = (points - matrices[camera, :3, 3]) @ matrices[camera, :3, :3]
                z = local[:, 2]
                k = intr_b[camera]
                u = torch.round(k[0, 0] * local[:, 0] / z.clamp_min(1e-6) + k[0, 2]).long()
                v = torch.round(k[1, 1] * local[:, 1] / z.clamp_min(1e-6) + k[1, 2]).long()
                inside = (z > 0) & (u >= 0) & (u < image_shape[1]) & (v >= 0) & (v < image_shape[0])
                if bool(inside.any().item()):
                    linear = v[inside] * image_shape[1] + u[inside]
                    zbuf = torch.full((image_shape[0] * image_shape[1],), torch.inf, device=depth.device)
                    zbuf.scatter_reduce_(0, linear, z[inside], reduce="amin", include_self=True)
                    winner = torch.isclose(z[inside], zbuf[linear], atol=1e-6)
                    out = color[camera].reshape(-1, 3)
                    out[linear[winner]] = colors[inside][winner]
        return depth, normals, color[0] if unbatched else color, valid

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
        mesh = self.extract_mesh(refine_iterations, surface_only)
        observations = [texture_observations] if isinstance(texture_observations, CameraObservation) else list(texture_observations)
        if not observations or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return mesh
        points = torch.as_tensor(mesh.vertices, device=self.device, dtype=torch.float32)
        accum = torch.zeros((len(points), 3), device=self.device, dtype=torch.float32)
        counts = torch.zeros(len(points), device=self.device, dtype=torch.float32)
        minimum = self.config.depth_minimum_distance if camera_min_distance is None else float(camera_min_distance)
        maximum = self.config.depth_maximum_distance if camera_max_distance is None else float(camera_max_distance)
        tolerance = (2.0 * self.config.voxel_size if texture_depth_tolerance_m is None
                     else float(texture_depth_tolerance_m))
        for obs in observations:
            if obs.rgb_image is None or obs.intrinsics is None or obs.pose is None:
                raise ValueError("texture observations require rgb_image, intrinsics, and pose")
            rgb = obs.rgb_image.to(self.device)
            depth = obs.depth_image.to(self.device) if obs.depth_image is not None else None
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            intr = obs.intrinsics.to(self.device, dtype=torch.float32)
            if intr.ndim == 2:
                intr = intr.unsqueeze(0)
            matrices = obs.pose.get_matrix().to(self.device, dtype=torch.float32)
            if matrices.ndim == 2:
                matrices = matrices.unsqueeze(0)
            if depth is not None and depth.ndim == 2:
                depth = depth.unsqueeze(0)
            for camera in range(min(len(rgb), len(intr), len(matrices))):
                local = (points - matrices[camera, :3, 3]) @ matrices[camera, :3, :3]
                z = local[:, 2]
                u = torch.round(intr[camera, 0, 0] * local[:, 0] / z.clamp_min(1e-6) + intr[camera, 0, 2]).long()
                v = torch.round(intr[camera, 1, 1] * local[:, 1] / z.clamp_min(1e-6) + intr[camera, 1, 2]).long()
                h, w = rgb.shape[1:3]
                good = (z >= minimum) & (z <= maximum) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if depth is not None:
                    sampled = depth[min(camera, len(depth)-1), v.clamp(0, h-1), u.clamp(0, w-1)]
                    good &= torch.isfinite(sampled) & ((sampled-z).abs() <= tolerance)
                if bool(good.any().item()):
                    accum[good] += rgb[camera, v[good], u[good]].float() / 255.0
                    counts[good] += 1
        if bool((counts > 0).any().item()):
            colors = torch.zeros_like(accum)
            colors[counts > 0] = accum[counts > 0] / counts[counts > 0, None]
            mesh.vertex_colors = colors
        return mesh

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> MatchedVoxels:
        data = self._portable_sparse.data
        if not data.has_features:
            raise RuntimeError("extract_matching_feature_voxels() requires feature_dim > 0")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be positive")
        query = torch.as_tensor(feature_vector, device=self.device, dtype=torch.float32)
        if query.ndim != 1:
            raise ValueError("feature_vector must be a vector")
        pools = sorted(self._portable_sparse._pool_to_coord)
        if not pools:
            empty = self._extract_sparse_occupied_voxels(surface_only=surface_only, sdf_threshold=sdf_threshold,
                                                        subvoxel_factor=1, max_points=None)
            return MatchedVoxels(empty, torch.empty(0, dtype=torch.int32, device=self.device),
                                 torch.empty(0, dtype=torch.float32, device=self.device))
        pidx = torch.as_tensor(pools, device=self.device, dtype=torch.long)
        sums = data.block_features[pidx].float().sum(1)
        weights = data.block_feature_weight[pidx].float().sum(1).clamp_min(1e-6)
        descriptors = sums / weights[:, None]
        if feature_projector is not None:
            descriptors = torch.as_tensor(feature_projector(descriptors), device=self.device, dtype=torch.float32)
        if query.numel() != descriptors.shape[1]:
            raise ValueError("feature_vector dimension does not match stored features")
        scores = torch.nn.functional.normalize(descriptors, dim=1) @ torch.nn.functional.normalize(query, dim=0)
        values, order = torch.topk(scores, min(top_k, len(scores)), sorted=True)
        if minimum_score is not None:
            keep = values >= float(minimum_score)
            values, order = values[keep], order[keep]
        selected_pool = pidx[order].to(torch.int32)
        voxels = self._extract_sparse_occupied_voxels(surface_only=surface_only, sdf_threshold=sdf_threshold,
                                                      subvoxel_factor=1, max_points=None)
        keep_voxel = torch.isin(voxels.block_idx_per_voxel.to(torch.long), selected_pool.to(torch.long))
        voxels = OccupiedVoxels(voxels.centers[keep_voxel], voxels.block_idx_per_voxel[keep_voxel], voxels.block_data,
                                subvoxel_factor=voxels.subvoxel_factor)
        return MatchedVoxels(voxels, selected_pool, values.float())

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
        # Clear targets the dynamic channel; static scene occupancy survives
        # a dynamic reset and remains exported through its static channel.
        mask &= ~self._static_mask
        count = int((state.weight[mask] > 0).sum().item())
        sparse = self._portable_sparse.data
        dynamic_pools = []
        static_pools = []
        for pool, coord in self._portable_sparse._pool_to_coord.items():
            grid_d, grid_h, grid_w = (int(value) for value in sparse.grid_shape)
            block_offsets = torch.tensor(
                [math.ceil(grid_w / sparse.block_size) // 2,
                 math.ceil(grid_h / sparse.block_size) // 2,
                 math.ceil(grid_d / sparse.block_size) // 2], device=self.device,
                dtype=lower.dtype,
            )
            center_offset = torch.tensor([grid_w, grid_h, grid_d], device=self.device,
                                         dtype=lower.dtype) * 0.5
            block_min = sparse.origin + (
                (torch.as_tensor(coord, device=self.device, dtype=lower.dtype) + block_offsets)
                * sparse.block_size - center_offset
            ) * sparse.voxel_size
            block_max = block_min + sparse.block_size * sparse.voxel_size
            if bool(((block_max >= lower) & (block_min <= upper)).all().item()):
                dynamic_pools.append(pool)
                if sparse.has_static and bool(torch.isfinite(sparse.static_block_data[pool]).any().item()):
                    static_pools.append(pool)
        if static_pools:
            # Preserve static-only blocks while removing dynamic/appearance
            # payload from mixed blocks.
            for pool in static_pools:
                sparse.block_data[pool].zero_()
                sparse.block_grid_rgb[pool].zero_()
                if sparse.has_features:
                    sparse.block_features[pool].zero_()
                    sparse.block_feature_weight[pool].zero_()
            dynamic_pools = [pool for pool in dynamic_pools if pool not in static_pools]
        sparse_cleared = self._portable_sparse.clear_blocks(dynamic_pools)
        tsdf = torch.where(mask, torch.ones_like(state.tsdf), state.tsdf)
        weight = torch.where(mask, torch.zeros_like(state.weight), state.weight)
        occupancy = state.occupancy & ~mask
        esdf, gradient = dense_esdf(occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
                                    dtype=state.tsdf.dtype)
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy, esdf=esdf,
                            gradient=gradient, generation=state.generation + (mask.any() | bool(sparse_cleared)).to(torch.int64))
        if count or sparse_cleared:
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
        self._sync_sparse_static_layer(mask)
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy,
                            esdf=state.esdf, gradient=state.gradient,
                            generation=state.generation + 1)
        self._apply_static_layer()
        self._invalidate_esdf_cache()
        return int(mask.sum().item())

    def _sync_sparse_static_layer(self, mask: torch.Tensor) -> None:
        """Mirror dense static occupancy into the exported sparse channel."""
        data = self._portable_sparse.data
        if not data.has_static:
            return
        bs = int(data.block_size)

        # A static-scene update is a replacement, rather than an additive
        # operation.  ``_strip_static_layer`` only removes the dense overlay;
        # the sparse pool has to be reconciled separately.  In particular,
        # leaving an old static-only block in the pool makes it continue to be
        # exported and prevents its slot from being reused by the replacement
        # scene.  Mixed blocks retain their dynamic/appearance payload and
        # only lose the old static channel.
        old_static_only: list[int] = []
        for pool in sorted(self._portable_sparse._pool_to_coord):
            static_values = data.static_block_data[pool]
            if not bool(torch.isfinite(static_values).any().item()):
                continue
            dynamic = bool((data.block_data[pool, :, 1] > 0).any().item())
            dynamic = dynamic or bool((data.block_grid_rgb[pool, :, 3] > 0).any().item())
            if data.has_features:
                dynamic = dynamic or bool((data.block_feature_weight[pool] > 0).any().item())
            if dynamic:
                data.static_block_data[pool].fill_(float("inf"))
                data.static_block_sums[pool] = 0
            else:
                old_static_only.append(pool)
        if old_static_only:
            self._portable_sparse.clear_blocks(old_static_only)

        occupied = torch.nonzero(mask, as_tuple=False)
        # Dense state uses native x,y,z order, matching sparse block keys.
        grid_d, grid_h, grid_w = (int(value) for value in data.grid_shape)
        block_offsets = torch.tensor(
            [math.ceil(grid_w / bs) // 2, math.ceil(grid_h / bs) // 2,
             math.ceil(grid_d / bs) // 2], device=occupied.device, dtype=torch.long
        )
        block_keys = (torch.div(occupied, bs, rounding_mode="floor") - block_offsets)
        block_keys = torch.unique(block_keys, dim=0) if occupied.numel() else occupied.new_empty((0, 3))
        for key in block_keys.detach().cpu().tolist():
            pool, _ = self._portable_sparse._allocate(tuple(int(v) for v in key))
            if pool is None:
                continue
            data.static_block_data[pool].fill_(float("inf"))
            data.static_block_sums[pool] = 0
        for pool, key in list(self._portable_sparse._pool_to_coord.items()):
            base = (torch.as_tensor(key, device=self.device, dtype=torch.long) + block_offsets) * bs
            values = []
            for lx in range(bs):
                for ly in range(bs):
                    for lz in range(bs):
                        index = base + torch.tensor((lx, ly, lz), device=self.device)
                        if bool((index < torch.tensor(mask.shape, device=self.device)).all()) and bool(mask[index[0], index[1], index[2]].item()):
                            values.append((lx * bs + ly) * bs + lz)
            if values:
                data.static_block_data[pool, torch.as_tensor(values, device=self.device)] = -0.5
                data.static_block_sums[pool] = float(len(values))


# Dense-only convenience surface retained at runtime without adding public
# methods to the pinned Mapper declaration contract.
Mapper.device = Mapper._portable_device
Mapper.voxel_size = Mapper._portable_voxel_size
Mapper.is_esdf_current = Mapper._portable_is_esdf_current
Mapper.get_voxel_grid = Mapper._portable_get_voxel_grid
Mapper.query = Mapper._portable_query
