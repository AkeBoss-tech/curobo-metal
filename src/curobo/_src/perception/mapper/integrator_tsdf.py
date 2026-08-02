"""Portable high-level TSDF integrator backed by :class:`Mapper`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from .mapper import Mapper
from .mapper_cfg import MapperCfg


@dataclass
class BlockSparseTSDFIntegratorCfg:
    voxel_size: float = 0.005
    origin: torch.Tensor | None = None
    truncation_distance: float = 0.04
    max_blocks: Optional[int] = None
    hash_capacity: Optional[int] = None
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 1.0
    time_decay: float = 1.0
    minimum_tsdf_weight: float = 0.1
    grid_shape: Tuple[int, int, int] | None = None
    roughness: float = 3.0
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    num_cameras: int = 1
    device: str = "cuda:0"
    block_size: int = 8
    feature_dim: int = 0
    color_grid_size: int = 1
    accumulator_w_max: float = 1000.0

    def __post_init__(self):
        if self.grid_shape is None:
            raise ValueError("grid_shape is required by the portable bounded mapper")
        if self.feature_dim:
            raise NotImplementedError("feature-volume integration requires Warp/CUDA")
        if self.origin is None:
            self.origin = torch.zeros(3)


class BlockSparseTSDFIntegrator:
    """High-level TSDF facade backed by the portable dense Mapper.

    This accepts the familiar block-sparse configuration for source
    compatibility, but intentionally exposes a dense tensor state rather than
    pretending to expose Warp block-pool pointers.
    """

    def __init__(self, config: BlockSparseTSDFIntegratorCfg, kernels=None):
        if kernels is not None:
            raise NotImplementedError("custom Warp block-sparse kernels are unavailable on CPU/MPS")
        self.cfg = config
        extent = tuple(int(n) * self.cfg.voxel_size for n in self.cfg.grid_shape)
        center = torch.as_tensor(self.cfg.origin) + torch.as_tensor(extent) / 2
        self.mapper = Mapper(MapperCfg(
            extent_meters_xyz=extent,
            voxel_size=self.cfg.voxel_size,
            grid_center=center,
            truncation_distance=self.cfg.truncation_distance,
            minimum_tsdf_weight=self.cfg.minimum_tsdf_weight,
            depth_minimum_distance=self.cfg.depth_minimum_distance,
            depth_maximum_distance=self.cfg.depth_maximum_distance,
            block_size=self.cfg.block_size,
            device=self.cfg.device,
            accumulator_w_max=self.cfg.accumulator_w_max,
        ))

    @property
    def tsdf(self):
        return self.mapper._mapper.state

    def integrate(self, observation=None, camera_observation=None, lidar_observation=None):
        return self.mapper.integrate(
            observation=observation, camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )

    def extract_mesh(self, refine_iterations=0, surface_only=False, level=0.0):
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        mesh = self.mapper.extract_mesh(refine_iterations, surface_only)
        return mesh.vertices, mesh.faces, torch.zeros_like(mesh.vertices)

    def extract_mesh_tensors(self, level=0.0, surface_only=False, refine_iterations=0):
        return self.extract_mesh(refine_iterations, surface_only, level)

    def extract_occupied_voxels(self, surface_only=False, sdf_threshold=None, subvoxel_factor=1,
                                max_points=None, texture_observations=None,
                                camera_min_distance=None, camera_max_distance=None,
                                texture_depth_tolerance_m=None):
        return self.mapper.extract_occupied_voxels(
            surface_only, sdf_threshold, subvoxel_factor=subvoxel_factor, max_points=max_points,
            texture_observations=texture_observations, camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance, texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

    def extract_surface_voxels(self, sdf_threshold=None):
        return self.extract_occupied_voxels(True, sdf_threshold)

    def extract_textured_mesh(self, texture_observations, refine_iterations=0, surface_only=True,
                              level=0.0, camera_min_distance=None, camera_max_distance=None,
                              texture_depth_tolerance_m=None):
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        return self.mapper.extract_textured_mesh(
            texture_observations, refine_iterations, surface_only, camera_min_distance,
            camera_max_distance, texture_depth_tolerance_m,
        )

    def render(self, intrinsics, pose, image_shape):
        return self.mapper.render(intrinsics, pose, image_shape)

    def reset(self):
        return self.mapper.reset()

    def save_blocks(self, path):
        return self.mapper.save_blocks(path)

    def import_blocks(self, blocks):
        if isinstance(blocks, (str, bytes)):
            return self.mapper.import_blocks(blocks)
        if not isinstance(blocks, dict):
            raise TypeError("blocks must be a state dictionary or checkpoint path")
        self.mapper._mapper.load_state_dict(blocks)

    def clear_region(self, bounds_min, bounds_max):
        return self.mapper.clear_region(bounds_min, bounds_max)

    def clear_blocks(self, pool_indices):
        return self.mapper.clear_blocks(pool_indices)

    def recycle_empty_blocks(self):
        # Dense maps have no unused block pool; return a stable no-op count.
        return 0

    def get_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                    surface_only=False, sdf_threshold=None, minimum_score=None,
                                    feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        return self.mapper.get_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector
        )

    extract_matching_feature_voxels = get_matching_feature_voxels

    def update_static_obstacles(self, scene, env_idx=0, debug=False):
        del debug
        return self.mapper.update_static_obstacles(scene, env_idx)

    def get_stats(self):
        return self.mapper.get_stats()
