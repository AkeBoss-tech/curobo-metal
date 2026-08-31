"""Canonical coordinate transforms for volumetric mapping."""

from typing import Tuple, Union

import torch


def voxel_to_world(
    iz: Union[int, float, torch.Tensor],
    iy: Union[int, float, torch.Tensor],
    ix: Union[int, float, torch.Tensor],
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center
    world_x = cx + (float(ix) - (nx - 1) / 2.0) * s
    world_y = cy + (float(iy) - (ny - 1) / 2.0) * s
    world_z = cz + (float(iz) - (nz - 1) / 2.0) * s
    return (world_x, world_y, world_z)


def world_to_voxel(
    world_x: float,
    world_y: float,
    world_z: float,
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[int, int, int]:
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center
    fx = (world_x - cx) / s + (nx - 1) / 2.0
    fy = (world_y - cy) / s + (ny - 1) / 2.0
    fz = (world_z - cz) / s + (nz - 1) / 2.0
    ix = int(round(fx))
    iy = int(round(fy))
    iz = int(round(fz))
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        return (iz, iy, ix)
    return (-1, -1, -1)


def world_to_voxel_continuous(
    world_x: float,
    world_y: float,
    world_z: float,
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center
    fx = (world_x - cx) / s + (nx - 1) / 2.0
    fy = (world_y - cy) / s + (ny - 1) / 2.0
    fz = (world_z - cz) / s + (nz - 1) / 2.0
    return (fz, fy, fx)


def get_grid_bounds(
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center
    half_extent_x = (nx - 1) / 2.0 * s + s / 2.0
    half_extent_y = (ny - 1) / 2.0 * s + s / 2.0
    half_extent_z = (nz - 1) / 2.0 * s + s / 2.0
    min_corner = (cx - half_extent_x, cy - half_extent_y, cz - half_extent_z)
    max_corner = (cx + half_extent_x, cy + half_extent_y, cz + half_extent_z)
    return (min_corner, max_corner)


def get_grid_extent(
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    nz, ny, nx = grid_shape
    return (nx * voxel_size, ny * voxel_size, nz * voxel_size)
