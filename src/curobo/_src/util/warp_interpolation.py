"""Compatibility boundary for the former Warp interpolation module."""

from curobo._src.util.trajectory import get_cpu_linear_interpolation


def get_cuda_linear_interpolation(raw_traj, traj_tsteps, out_traj):
    """Portable torch implementation retaining the historical CUDA-named API."""
    return get_cpu_linear_interpolation(raw_traj, traj_tsteps, out_traj, kind=None)


def linear_interpolate_batch_dt_trajectory_kernel(*args, **kwargs):
    raise NotImplementedError(
        "the raw Warp kernel is CUDA-only; call get_cuda_linear_interpolation instead"
    )


__all__ = ["get_cuda_linear_interpolation", "linear_interpolate_batch_dt_trajectory_kernel"]
