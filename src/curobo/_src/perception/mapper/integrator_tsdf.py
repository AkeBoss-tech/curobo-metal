"""Portable high-level TSDF integrator backed by :class:`Mapper`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    def __init__(self, cfg: BlockSparseTSDFIntegratorCfg | None = None, **kwargs):
        self.cfg = cfg or BlockSparseTSDFIntegratorCfg(**kwargs)
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

    def integrate(self, observation, *args, **kwargs):
        return self.mapper.integrate(observation, *args, **kwargs)

    def extract_mesh(self, *args, **kwargs):
        mesh = self.mapper.extract_mesh(*args, **kwargs)
        return mesh.vertices, mesh.faces, torch.zeros_like(mesh.vertices)

    def extract_occupied_voxels(self, *args, **kwargs):
        return self.mapper.extract_occupied_voxels(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.mapper.render(*args, **kwargs)

    def reset(self):
        return self.mapper.reset()

    def save_blocks(self, path):
        return self.mapper.save_blocks(path)

    def import_blocks(self, path, import_weight=None):
        return self.mapper.import_blocks(path, import_weight)

    def get_stats(self):
        return self.mapper.get_stats()
