"""Compatibility boundary for the former Warp interpolation module."""

from __future__ import annotations

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.util.warp import get_warp_device_stream, init_warp


class _PortableWarpDeclarations:
    @staticmethod
    def kernel(function):
        return function


wp = _PortableWarpDeclarations()


def get_cuda_linear_interpolation(
    raw_traj: JointState, traj_tsteps: torch.Tensor, out_traj: JointState
):
    """Portable torch implementation retaining the historical CUDA-named API."""
    # Import lazily to avoid a module-initialization cycle with trajectory.py.
    from curobo._src.util.trajectory import get_cuda_linear_interpolation as _portable

    return _portable(raw_traj, traj_tsteps, out_traj)


@wp.kernel
def linear_interpolate_batch_dt_trajectory_kernel(
    raw_position: wp.array(dtype=wp.float32),
    raw_velocity: wp.array(dtype=wp.float32),
    raw_acceleration: wp.array(dtype=wp.float32),
    raw_jerk: wp.array(dtype=wp.float32),
    raw_dt: wp.array(dtype=wp.float32),
    out_position: wp.array(dtype=wp.float32),
    out_velocity: wp.array(dtype=wp.float32),
    out_acceleration: wp.array(dtype=wp.float32),
    out_jerk: wp.array(dtype=wp.float32),
    out_dt: wp.array(dtype=wp.float32),
    traj_tsteps: wp.array(dtype=wp.int32),
    batch_size: wp.int32,
    raw_horizon: wp.int32,
    dof: wp.int32,
    out_horizon: wp.int32,
):
    del (
        raw_position,
        raw_velocity,
        raw_acceleration,
        raw_jerk,
        raw_dt,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        out_dt,
        traj_tsteps,
        batch_size,
        raw_horizon,
        dof,
        out_horizon,
    )
    raise NotImplementedError(
        "the raw Warp kernel is CUDA-only; call get_cuda_linear_interpolation instead"
    )


__all__ = ["get_cuda_linear_interpolation", "linear_interpolate_batch_dt_trajectory_kernel"]
