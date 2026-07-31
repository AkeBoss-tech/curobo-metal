"""Coordinate conversions using the pinned center-aligned voxel convention."""

from __future__ import annotations

import torch


def get_grid_extent(grid_shape, voxel_size):
    return tuple(float(v) * float(voxel_size) for v in grid_shape)


def get_grid_bounds(grid_center, grid_shape, voxel_size):
    extent = torch.as_tensor(get_grid_extent(grid_shape, voxel_size), dtype=torch.float64)
    center = torch.as_tensor(grid_center, dtype=torch.float64)
    return tuple((center - extent / 2).tolist()), tuple((center + extent / 2).tolist())


def voxel_to_world(iz, iy, ix, grid_center, grid_shape, voxel_size):
    origin = torch.as_tensor(get_grid_bounds(grid_center, grid_shape, voxel_size)[0])
    xyz = origin + (torch.stack([torch.as_tensor(ix), torch.as_tensor(iy), torch.as_tensor(iz)]) + 0.5) * voxel_size
    return tuple(xyz.tolist()) if not any(isinstance(v, torch.Tensor) for v in (iz, iy, ix)) else xyz


def world_to_voxel_continuous(world_x, world_y, world_z, grid_center, grid_shape, voxel_size):
    origin = torch.as_tensor(get_grid_bounds(grid_center, grid_shape, voxel_size)[0])
    xyz = (torch.as_tensor([world_x, world_y, world_z]) - origin) / voxel_size - 0.5
    return xyz[2], xyz[1], xyz[0]


def world_to_voxel(world_x, world_y, world_z, grid_center, grid_shape, voxel_size):
    values = world_to_voxel_continuous(
        world_x, world_y, world_z, grid_center, grid_shape, voxel_size
    )
    return tuple(int(torch.floor(v).item()) for v in values)
