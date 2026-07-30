from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch


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
        if len(self.extent_meters_xyz) != 3 or any(x <= 0 for x in self.extent_meters_xyz):
            raise ValueError("extent_meters_xyz must contain three positive values")
        if self.voxel_size <= 0 or self.truncation_distance <= 0:
            raise ValueError("voxel_size and truncation_distance must be positive")

    @property
    def grid_shape(self):
        return tuple(max(2, int(math.ceil(x/self.voxel_size))) for x in self.extent_meters_xyz)

    @property
    def origin(self):
        center = torch.zeros(3) if self.grid_center is None else torch.as_tensor(self.grid_center)
        return center - torch.tensor(self.get_actual_extent()) / 2

    def get_actual_extent(self):
        return tuple(x*self.voxel_size for x in self.grid_shape)

    def voxel_to_world(self, iz, iy, ix):
        return tuple((self.origin + torch.tensor([ix,iy,iz])*self.voxel_size).tolist())

    def world_to_voxel(self, world_x, world_y, world_z):
        xyz = torch.floor((torch.tensor([world_x,world_y,world_z])-self.origin)/self.voxel_size).long()
        return int(xyz[2]), int(xyz[1]), int(xyz[0])

    def get_grid_bounds(self):
        origin = self.origin
        return tuple(origin.tolist()), tuple((origin+torch.tensor(self.get_actual_extent())).tolist())
