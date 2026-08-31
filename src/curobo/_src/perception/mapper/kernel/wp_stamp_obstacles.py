from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Tuple, Union

import torch

from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES
from curobo._src.types.pose import Pose
from . import raw_kernel

annotations = check_int32_tensors = get_warp_device_stream = None
compute_local_sdf = is_obs_enabled = load_obstacle_transform = raw_kernel
wp = None


def compute_aabb_block_bounds(
    dims: torch.Tensor,
    inv_pose: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    truncation: float,
    grid_dims: Tuple[int, int, int],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return raw_kernel(dims, inv_pose, origin, voxel_size, truncation, grid_dims, block_size)


def stamp_scene_obstacles(
    tsdf: "BlockSparseTSDF",
    scene: "SceneData",
    env_idx: int = 0,
    static_color: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    debug: bool = True,
) -> None:
    return raw_kernel(tsdf, scene, env_idx, static_color, debug)


def stamp_obstacles(
    tsdf: "BlockSparseTSDF",
    obs_data: Union["CuboidData", "MeshData", "VoxelData"],
    env_idx: int,
    static_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    debug: bool = False,
) -> None:
    return raw_kernel(tsdf, obs_data, env_idx, static_color, debug)


def clear_static_channel(tsdf_data: "BlockSparseTSDFData") -> None:
    if hasattr(tsdf_data, "static"):
        tsdf_data.static.zero_()
    return tsdf_data
