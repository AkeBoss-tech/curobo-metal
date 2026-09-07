from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch

from curobo._src.perception.mapper.block_allocation import calculate_tsdf_max_blocks
from curobo._src.perception.mapper.constants import (
    _validate_color_grid_size,
    _validate_feature_block_grid_size,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
)
from curobo._src.util.logging import log_and_raise


@dataclass
class MapperCfg:
    extent_meters_xyz: Tuple[float,float,float]
    voxel_size: float = 0.005
    esdf_voxel_size: float = 0.05
    extent_esdf_meters_xyz: Tuple[float,float,float] | None = None
    grid_center: Optional[torch.Tensor] = None
    truncation_distance: float = 0.04
    minimum_tsdf_weight: float = 0.1
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    decay_factor: float = 1.0
    frustum_decay_factor: float = 1.0
    block_size: int = 8
    hash_load_factor: float = 0.5
    roughness: float = 3.0
    color_grid_size: int = 1
    seeding_method: str = "gather"
    edt_solver: str = "pba"
    enable_static: bool = False
    static_obstacle_color: Tuple[int,int,int] = (20,20,20)
    num_cameras: int = 1
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
    max_support_pixels_per_block_lidar: int = 8
    feature_dim: int = 0
    feature_block_grid_size: int = 1
    feature_grid_height: Optional[int] = None
    feature_grid_width: Optional[int] = None
    max_visible_blocks_per_integration: Optional[int] = None
    max_support_pixels_per_block_camera: int = 8
    feature_channels_per_thread: int = 8
    max_feature_tile_channels: int = 4096
    feature_integration_kernel: str = "auto"
    profile_integration_kernel_timings: bool = False
    accumulator_w_max: float = 1000.0
    device: str = "cuda:0"

    def __post_init__(self):
        if len(self.extent_meters_xyz) != 3 or any(not math.isfinite(float(x)) or x <= 0 for x in self.extent_meters_xyz):
            raise ValueError("extent_meters_xyz must contain three positive values")
        values = (self.voxel_size, self.esdf_voxel_size, self.truncation_distance,
                  self.minimum_tsdf_weight, self.depth_minimum_distance,
                  self.depth_maximum_distance, self.decay_factor,
                  self.frustum_decay_factor, self.hash_load_factor, self.roughness,
                  self.accumulator_w_max)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("MapperCfg scalar values must be finite")
        if self.voxel_size <= 0 or self.esdf_voxel_size <= 0 or self.truncation_distance <= 0:
            raise ValueError("voxel_size, esdf_voxel_size, and truncation_distance must be positive")
        if self.minimum_tsdf_weight < 0 or self.accumulator_w_max <= 0 or self.roughness <= 0:
            raise ValueError("minimum_tsdf_weight must be nonnegative and accumulator_w_max/roughness positive")
        if self.depth_minimum_distance < 0 or self.depth_minimum_distance >= self.depth_maximum_distance:
            raise ValueError("depth_minimum_distance must be nonnegative and less than depth_maximum_distance")
        if not 0 < self.decay_factor <= 1 or not 0 < self.frustum_decay_factor <= 1:
            raise ValueError("decay_factor and frustum_decay_factor must be in (0, 1]")
        if not 0 < self.hash_load_factor <= 1:
            raise ValueError("hash_load_factor must be in (0, 1]")
        if not isinstance(self.block_size, int) or self.block_size < 1 or self.block_size > 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be 1 or a power of two no greater than 32")
        if self.seeding_method not in {"gather", "scatter"}:
            raise ValueError("seeding_method must be 'gather' or 'scatter'")
        if self.edt_solver not in {"pba", "jfa"}:
            raise ValueError("edt_solver must be 'pba' or 'jfa'")
        if self.feature_integration_kernel not in {"auto", "grouped", "tiled"}:
            raise ValueError("feature_integration_kernel must be 'auto', 'grouped', or 'tiled'")
        if not isinstance(self.profile_integration_kernel_timings, bool):
            raise TypeError("profile_integration_kernel_timings must be bool")
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be positive")
        if self.image_height is not None or self.image_width is not None:
            if self.image_height is None or self.image_width is None:
                raise ValueError("image_height and image_width must be specified together")
            if self.image_height <= 0 or self.image_width <= 0:
                raise ValueError("image_height and image_width must be positive")
        if (self.texture_camera_image_height is None) != (self.texture_camera_image_width is None):
            raise ValueError("texture_camera_image_height and texture_camera_image_width must be specified together")
        if self.texture_num_cameras is None:
            self.texture_num_cameras = self.num_cameras
        if self.texture_num_cameras <= 0:
            raise ValueError("texture_num_cameras must be positive")
        if self.texture_camera_image_height is None:
            self.texture_camera_image_height = self.image_height
            self.texture_camera_image_width = self.image_width
        elif self.texture_camera_image_height <= 0 or self.texture_camera_image_width <= 0:
            raise ValueError("texture camera dimensions must be positive")
        if self.feature_dim < 0 or self.feature_block_grid_size < 1 or self.feature_channels_per_thread < 1:
            raise ValueError("feature dimensions must be nonnegative/positive as appropriate")
        _validate_color_grid_size(self.color_grid_size, self.block_size)
        _validate_feature_block_grid_size(self.feature_block_grid_size, self.block_size)
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim, self.feature_grid_height, self.feature_grid_width
        )
        if self.max_feature_tile_channels <= 0 or self.max_support_pixels_per_block_camera <= 0:
            raise ValueError("feature and camera support capacities must be positive")
        if self.lidar_num_sensors < 0 or self.max_support_pixels_per_block_lidar <= 0:
            raise ValueError("lidar_num_sensors must be nonnegative and lidar support capacity positive")
        if self.extent_esdf_meters_xyz is not None:
            if len(self.extent_esdf_meters_xyz) != 3 or any(not math.isfinite(float(x)) or x <= 0 for x in self.extent_esdf_meters_xyz):
                raise ValueError("extent_esdf_meters_xyz must contain three finite positive values")
        if self.grid_center is None:
            self.grid_center = torch.zeros(3, dtype=torch.float32)
        else:
            self.grid_center = torch.as_tensor(self.grid_center, dtype=torch.float32)
            if self.grid_center.shape != (3,) or not bool(torch.isfinite(self.grid_center).all().item()):
                raise ValueError("grid_center must be a finite xyz vector")

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Pinned V2 public order: ``(nz, ny, nx)``."""
        x, y, z = self.extent_meters_xyz
        return (max(2, int(math.ceil(z / self.voxel_size))),
                max(2, int(math.ceil(y / self.voxel_size))),
                max(2, int(math.ceil(x / self.voxel_size))))

    @property
    def _native_grid_shape_portable(self):
        """Dense CPU/MPS tensor shape in world ``(x, y, z)`` order."""
        nz, ny, nx = self.grid_shape
        return nx, ny, nz

    @property
    def max_blocks(self) -> int:
        return calculate_tsdf_max_blocks(
            self.grid_shape, self.voxel_size, self.block_size,
            self.truncation_distance, self.roughness,
        )

    @property
    def hash_capacity(self) -> int:
        # Dense storage does not use a hash table, but callers use this value
        # for memory planning.  Keep the same positive sizing relationship.
        return max(1, int(math.ceil(self.max_blocks / self.hash_load_factor)))

    @property
    def _origin_portable(self):
        return self.grid_center - torch.tensor(self.get_actual_extent(), dtype=self.grid_center.dtype) / 2

    def get_actual_extent(self) -> Tuple[float, float, float]:
        nz, ny, nx = self.grid_shape
        return nx * self.voxel_size, ny * self.voxel_size, nz * self.voxel_size

    def voxel_to_world(
        self, iz: int, iy: int, ix: int
    ) -> Tuple[float, float, float]:
        nz, ny, nx = self.grid_shape
        cx, cy, cz = self.grid_center.tolist()
        return (cx + (int(ix) - (nx - 1) / 2.0) * self.voxel_size,
                cy + (int(iy) - (ny - 1) / 2.0) * self.voxel_size,
                cz + (int(iz) - (nz - 1) / 2.0) * self.voxel_size)

    def world_to_voxel(
        self, world_x: float, world_y: float, world_z: float
    ) -> Tuple[int, int, int]:
        nz, ny, nx = self.grid_shape
        cx, cy, cz = self.grid_center.tolist()
        ix = int(round((float(world_x) - cx) / self.voxel_size + (nx - 1) / 2.0))
        iy = int(round((float(world_y) - cy) / self.voxel_size + (ny - 1) / 2.0))
        iz = int(round((float(world_z) - cz) / self.voxel_size + (nz - 1) / 2.0))
        if 0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx:
            return iz, iy, ix
        return -1, -1, -1

    def get_grid_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        origin = self.origin
        return tuple(origin.tolist()), tuple((origin+torch.tensor(self.get_actual_extent())).tolist())


# Dense portable mapping additionally exposes native tensor-order and origin
# helpers.  Install them from private properties so the declared mapper config
# remains the pinned V2 shape.
MapperCfg.native_grid_shape = MapperCfg.__dict__["_native_grid_shape_portable"]
MapperCfg.origin = MapperCfg.__dict__["_origin_portable"]
