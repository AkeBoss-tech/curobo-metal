"""Portable dense ESDF integration with the cuRobo block-sparse facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from .integrator_tsdf import BlockSparseTSDFIntegrator, BlockSparseTSDFIntegratorCfg


@dataclass
class BlockSparseESDFIntegratorCfg:
    voxel_size: float = 0.005
    origin: torch.Tensor | None = None
    esdf_voxel_size: Optional[float] = None
    esdf_grid_shape: Tuple[int, int, int] | None = None
    truncation_distance: float = 0.04
    max_blocks: Optional[int] = None
    hash_capacity: Optional[int] = None
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 0.5
    time_decay: float = 1.0
    minimum_tsdf_weight: float = 0.1
    blend_esdf: bool = False
    use_cuda_graph: bool = True
    grid_shape: Tuple[int, int, int] | None = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    texture_num_cameras: Optional[int] = None
    texture_camera_image_height: Optional[int] = None
    texture_camera_image_width: Optional[int] = None
    lidar_num_sensors: int = 0
    lidar_image_height: Optional[int] = None
    lidar_image_width: Optional[int] = None
    lidar_feature_grid_height: Optional[int] = None
    lidar_feature_grid_width: Optional[int] = None
    lidar_linear_interpolation_max_allowable_difference_vox: float = 2.0
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox: float = 0.5
    max_visible_blocks_per_lidar_integration: Optional[int] = None
    max_support_pixels_per_block_lidar: int = 32
    enable_static: bool = False
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    adjacent_skip_steps: float = 1.0
    seeding_method: str = "gather"
    edt_solver: str = "pba"
    roughness: float = 3.0
    num_cameras: int = 1
    block_size: int = 8
    feature_dim: int = 0
    feature_block_grid_size: int = 1
    feature_grid_height: Optional[int] = None
    feature_grid_width: Optional[int] = None
    max_visible_blocks_per_integration: Optional[int] = None
    max_support_pixels_per_block_camera: int = 32
    feature_channels_per_thread: int = 4
    max_feature_tile_channels: int = 4096
    color_grid_size: int = 1
    feature_integration_kernel: str = "auto"
    profile_integration_kernel_timings: bool = False
    accumulator_w_max: float = 1000.0

    def __post_init__(self) -> None:
        if self.esdf_voxel_size is None:
            self.esdf_voxel_size = self.voxel_size
        if self.esdf_voxel_size != self.voxel_size:
            raise NotImplementedError("portable ESDF resampling requires a dense resampling backend")
        if self.grid_shape is not None:
            self.grid_shape = tuple(int(n) for n in self.grid_shape)
            if len(self.grid_shape) != 3 or any(n <= 0 for n in self.grid_shape):
                raise ValueError("grid_shape must contain three positive dimensions")
        if self.esdf_grid_shape is None and self.grid_shape is not None:
            self.esdf_grid_shape = self.grid_shape
        if self.esdf_grid_shape is not None:
            self.esdf_grid_shape = tuple(int(n) for n in self.esdf_grid_shape)
        if self.grid_shape is not None and self.esdf_grid_shape != self.grid_shape:
            raise NotImplementedError("portable ESDF grid shape must match the bounded TSDF grid")
        if self.origin is None:
            self.origin = torch.zeros(3, dtype=torch.float32)
        else:
            self.origin = torch.as_tensor(self.origin, dtype=torch.float32)
        if self.origin.shape != (3,):
            raise ValueError("origin must be an xyz vector")
        if self.edt_solver not in {"pba", "jfa"}:
            raise ValueError("edt_solver must be 'pba' or 'jfa'")


class BlockSparseESDFIntegrator:
    """A CPU/MPS ESDF facade over :class:`BlockSparseTSDFIntegrator`.

    ``use_cuda_graph`` is accepted as a source-compatible request for a
    persistent execution state; raw CUDA graph handles are intentionally not
    exposed.  The exact dense ESDF is recomputed from the current dense TSDF.
    """

    def __init__(self, config: BlockSparseESDFIntegratorCfg):
        self.config = config
        self.cfg = config
        self.device = torch.device("mps" if config.device.startswith("cuda") and torch.backends.mps.is_available()
                                   else ("cpu" if config.device.startswith("cuda") else config.device))
        self.dtype = config.dtype
        self.use_cuda_graph = config.use_cuda_graph
        self._esdf_grid_shape = config.esdf_grid_shape
        self._esdf_voxel_size = torch.tensor([config.esdf_voxel_size], device=self.device, dtype=torch.float32)
        self._origin = config.origin.to(device=self.device)
        self._last_esdf_origin = self._origin.clone()
        self._frame_count = 0
        self._tsdf_integrator = None if config.grid_shape is None else BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
            voxel_size=config.voxel_size, origin=config.origin,
            truncation_distance=config.truncation_distance, max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity, depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance, frustum_decay=config.frustum_decay,
            time_decay=config.time_decay, minimum_tsdf_weight=config.minimum_tsdf_weight,
            grid_shape=config.grid_shape, roughness=config.roughness,
            image_height=config.image_height, image_width=config.image_width,
            texture_num_cameras=config.texture_num_cameras,
            texture_camera_image_height=config.texture_camera_image_height,
            texture_camera_image_width=config.texture_camera_image_width,
            lidar_num_sensors=config.lidar_num_sensors, lidar_image_height=config.lidar_image_height,
            lidar_image_width=config.lidar_image_width,
            lidar_feature_grid_height=config.lidar_feature_grid_height,
            lidar_feature_grid_width=config.lidar_feature_grid_width,
            lidar_linear_interpolation_max_allowable_difference_vox=config.lidar_linear_interpolation_max_allowable_difference_vox,
            lidar_nearest_interpolation_max_allowable_dist_to_ray_vox=config.lidar_nearest_interpolation_max_allowable_dist_to_ray_vox,
            max_visible_blocks_per_lidar_integration=config.max_visible_blocks_per_lidar_integration,
            max_support_pixels_per_block_lidar=config.max_support_pixels_per_block_lidar,
            enable_static=config.enable_static, static_obstacle_color=config.static_obstacle_color,
            seeding_method=config.seeding_method, num_cameras=config.num_cameras,
            device=config.device, block_size=config.block_size, feature_dim=config.feature_dim,
            feature_block_grid_size=config.feature_block_grid_size,
            feature_grid_height=config.feature_grid_height, feature_grid_width=config.feature_grid_width,
            max_visible_blocks_per_integration=config.max_visible_blocks_per_integration,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            feature_channels_per_thread=config.feature_channels_per_thread,
            max_feature_tile_channels=config.max_feature_tile_channels,
            color_grid_size=config.color_grid_size, feature_integration_kernel=config.feature_integration_kernel,
            profile_integration_kernel_timings=config.profile_integration_kernel_timings,
            accumulator_w_max=config.accumulator_w_max,
        ))
        self._dist_field = None if self._tsdf_integrator is None else self._tsdf_integrator.mapper._mapper.state.esdf[0]

    def _print_config(self):
        return {
            "tsdf_voxel_size": self.voxel_size,
            "esdf_voxel_size": self.esdf_voxel_size,
            "esdf_grid_shape": self.esdf_grid_shape,
            "persistent_execution": self.use_cuda_graph,
        }

    @property
    def tsdf(self):
        if self._tsdf_integrator is None:
            raise RuntimeError("tsdf requires grid_shape at construction")
        return self._tsdf_integrator.tsdf
    @property
    def esdf_grid_shape(self): return self._esdf_grid_shape
    @property
    def esdf_voxel_size(self): return float(self._esdf_voxel_size.item())
    @property
    def origin(self): return self._origin
    @property
    def voxel_size(self): return self.config.voxel_size
    @property
    def truncation_distance(self): return self.config.truncation_distance
    @property
    def grid_shape(self): return self.config.grid_shape
    @property
    def dist_field(self): return self._dist_field

    def reset(self):
        if self._tsdf_integrator is None:
            return None
        self._tsdf_integrator.reset()
        self._dist_field = self._tsdf_integrator.mapper._mapper.state.esdf[0]
        self._last_esdf_origin.copy_(self._origin)
        self._frame_count = 0

    def import_blocks(self, blocks):
        if self._tsdf_integrator is None:
            raise RuntimeError("import_blocks requires grid_shape at construction")
        result = self._tsdf_integrator.import_blocks(blocks)
        self._dist_field = self._tsdf_integrator.mapper._mapper.state.esdf[0]
        self._frame_count = int(result > 0)
        return result

    def integrate(self, observation=None, *, camera_observation=None, lidar_observation=None):
        if self._tsdf_integrator is None:
            raise RuntimeError("integrate requires grid_shape at construction")
        result = self._tsdf_integrator.integrate(
            observation=observation, camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        self._frame_count += 1
        return result

    def clear_region(self, bounds_min, bounds_max):
        if self._tsdf_integrator is None:
            raise RuntimeError("clear_region requires grid_shape at construction")
        return self._tsdf_integrator.clear_region(bounds_min, bounds_max)

    def clear_blocks(self, pool_indices):
        if self._tsdf_integrator is None:
            raise RuntimeError("clear_blocks requires grid_shape at construction")
        return self._tsdf_integrator.clear_blocks(pool_indices)

    def compute_esdf(self, esdf_origin: Optional[torch.Tensor] = None,
                     esdf_voxel_size: Optional[torch.Tensor] = None):
        if esdf_voxel_size is not None:
            value = float(torch.as_tensor(esdf_voxel_size).reshape(-1)[0].item())
            if value != self.config.voxel_size:
                raise NotImplementedError("portable ESDF resampling requires a dense resampling backend")
        if esdf_origin is not None:
            origin = torch.as_tensor(esdf_origin, device=self.device, dtype=self._origin.dtype)
            if origin.shape != (3,):
                raise ValueError("esdf_origin must be an xyz vector")
            if not torch.allclose(origin, self._origin):
                raise NotImplementedError("portable ESDF sliding windows are not implemented")
            self._last_esdf_origin.copy_(origin)
        if self._tsdf_integrator is None:
            raise RuntimeError("compute_esdf requires grid_shape at construction")
        self._dist_field = self._tsdf_integrator.mapper._mapper.state.esdf[0]
        return self._dist_field

    def _seed_esdf_impl(self, esdf_origin, esdf_voxel_size):
        return self.compute_esdf(esdf_origin, esdf_voxel_size)

    def _propagate_and_distance_impl(self, esdf_origin, esdf_voxel_size):
        return self.compute_esdf(esdf_origin, esdf_voxel_size)

    def _compute_esdf_impl(self, esdf_origin, esdf_voxel_size):
        return self.compute_esdf(esdf_origin, esdf_voxel_size)

    def compute(self, tsdf):
        """Historical convenience: compute exact ESDF for a compatible dense map."""
        if self._tsdf_integrator is None or (tsdf is not self and tsdf is not self._tsdf_integrator and tsdf is not self._tsdf_integrator.mapper):
            from ._portable import dense_state
            state = dense_state(tsdf)
            from curobo_metal.ops.perception import dense_esdf
            return dense_esdf(state.occupancy, self.config.voxel_size, 1.0, dtype=state.tsdf.dtype)
        field = self.compute_esdf()
        return field

    __call__ = compute

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = True, level: float = 0.0):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_mesh requires grid_shape at construction")
        return self._tsdf_integrator.extract_mesh(refine_iterations, surface_only, level)

    def extract_textured_mesh(self, texture_observations, refine_iterations: int = 0,
                              surface_only: bool = True, level: float = 0.0,
                              camera_min_distance: Optional[float] = None,
                              camera_max_distance: Optional[float] = None,
                              texture_depth_tolerance_m: Optional[float] = None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_textured_mesh requires grid_shape at construction")
        return self._tsdf_integrator.extract_textured_mesh(
            texture_observations, refine_iterations, surface_only, level,
            camera_min_distance, camera_max_distance, texture_depth_tolerance_m,
        )

    def extract_occupied_voxels(self, surface_only: bool = False, sdf_threshold: Optional[float] = None, *,
                                subvoxel_factor: int = 1, max_points: Optional[int] = None,
                                texture_observations=None, camera_min_distance=None,
                                camera_max_distance=None, texture_depth_tolerance_m=None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_occupied_voxels requires grid_shape at construction")
        return self._tsdf_integrator.extract_occupied_voxels(
            surface_only, sdf_threshold, subvoxel_factor=subvoxel_factor, max_points=max_points,
            texture_observations=texture_observations, camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance, texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_matching_feature_voxels requires grid_shape at construction")
        return self._tsdf_integrator.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector,
        )

    get_matching_feature_voxels = extract_matching_feature_voxels

    def get_voxel_grid(self):
        if self._tsdf_integrator is None:
            raise RuntimeError("get_voxel_grid requires grid_shape at construction")
        return self._tsdf_integrator.mapper.compute_esdf()

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False):
        if self._tsdf_integrator is None:
            return {"frame_count": 0, "storage": "external_dense_input"}
        stats = self._tsdf_integrator.get_stats(scan_pool, scan_hash)
        stats.update({"frame_count": self._frame_count, "esdf_grid_shape": self.esdf_grid_shape})
        return stats

    def memory_usage_mb(self):
        if self._tsdf_integrator is None:
            return 0.0
        return self._tsdf_integrator.memory_usage_mb()

    def update_static_obstacles(self, scene, env_idx: int = 0):
        if self._tsdf_integrator is None:
            raise RuntimeError("update_static_obstacles requires grid_shape at construction")
        return self._tsdf_integrator.update_static_obstacles(scene, env_idx)
