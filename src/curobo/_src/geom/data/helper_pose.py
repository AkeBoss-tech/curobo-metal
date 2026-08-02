"""Portable tensor equivalents of the small Warp pose-loading helpers."""
import torch
from ._portable import raw_warp


def _pose_row(inv_pose, idx):
    value = torch.as_tensor(inv_pose)
    if value.ndim < 2 or value.shape[-1] < 7:
        raise ValueError("inv_pose must have shape [N, >=7]")
    return value[idx]


def load_inv_position(inv_pose, idx):
    """Load the ``xyz`` inverse translation stored in a pose row."""
    return _pose_row(inv_pose, idx)[..., :3]


def load_inv_quat(inv_pose, idx):
    """Load the inverse quaternion in Warp's ``xyzw`` component order."""
    row = _pose_row(inv_pose, idx)
    return torch.stack((row[..., 4], row[..., 5], row[..., 6], row[..., 3]), dim=-1)


def get_forward_quat(inv_quat):
    """Return the conjugate of an inverse unit quaternion in ``xyzw`` order."""
    quat = torch.as_tensor(inv_quat)
    if quat.shape[-1:] != (4,):
        raise ValueError("inv_quat must end in dimension 4 (xyzw)")
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


get_obs_idx=load_transform_from_inv_pose=transform_point_to_local=transform_point_to_world=rotate_vector_to_world=raw_warp
__all__=["get_forward_quat","get_obs_idx","load_inv_position","load_inv_quat","load_transform_from_inv_pose","transform_point_to_local","transform_point_to_world","rotate_vector_to_world"]
