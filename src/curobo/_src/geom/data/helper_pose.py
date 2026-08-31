"""Portable tensor equivalents of the small Warp pose-loading helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from ._portable import raw_warp

wp = None


def _pose_row(inv_pose, idx):
    value = torch.as_tensor(inv_pose)
    if value.ndim < 2 or value.shape[-1] < 7:
        raise ValueError("inv_pose must have shape [N, >=7]")
    return value[idx]


def _load_inv_position_portable(inv_pose, idx):
    """Load the ``xyz`` inverse translation stored in a pose row."""
    return _pose_row(inv_pose, idx)[..., :3]


def _load_inv_quat_portable(inv_pose, idx):
    """Load the inverse quaternion in Warp's ``xyzw`` component order."""
    row = _pose_row(inv_pose, idx)
    return torch.stack((row[..., 4], row[..., 5], row[..., 6], row[..., 3]), dim=-1)


def _get_forward_quat_portable(inv_quat):
    """Return the conjugate of an inverse unit quaternion in ``xyzw`` order."""
    quat = torch.as_tensor(inv_quat)
    if quat.shape[-1:] != (4,):
        raise ValueError("inv_quat must end in dimension 4 (xyzw)")
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def _get_obs_idx_portable(env_idx, local_idx, max_n):
    return int(env_idx) * int(max_n) + int(local_idx)


def _load_transform_from_inv_pose_portable(inv_pose, flat_idx):
    return _pose_row(inv_pose, flat_idx)


def get_obs_idx(env_idx: wp.int32, local_idx: wp.int32, max_n: wp.int32) -> wp.int32:
    raise NotImplementedError


def load_inv_position(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.vec3:
    raise NotImplementedError


def load_inv_quat(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.quat:
    raise NotImplementedError


def get_forward_quat(inv_quat: wp.quat) -> wp.quat:
    raise NotImplementedError


def load_transform_from_inv_pose(inv_pose: wp.array2d(dtype=wp.float32), flat_idx: wp.int32) -> wp.transform:
    raise NotImplementedError


if not TYPE_CHECKING:
    get_obs_idx = _get_obs_idx_portable
    load_inv_position = _load_inv_position_portable
    load_inv_quat = _load_inv_quat_portable
    get_forward_quat = _get_forward_quat_portable
    load_transform_from_inv_pose = _load_transform_from_inv_pose_portable


transform_point_to_local = transform_point_to_world = rotate_vector_to_world = raw_warp
__all__=["get_forward_quat","get_obs_idx","load_inv_position","load_inv_quat","load_transform_from_inv_pose","transform_point_to_local","transform_point_to_world","rotate_vector_to_world"]
