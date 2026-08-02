"""Compatibility boundary for the former Warp interpolation module."""


def get_cuda_linear_interpolation(raw_traj, traj_tsteps, out_traj):
    """Portable torch implementation retaining the historical CUDA-named API."""
    # Import lazily to avoid a module-initialization cycle with trajectory.py.
    from curobo._src.util.trajectory import get_cuda_linear_interpolation as _portable

    return _portable(raw_traj, traj_tsteps, out_traj)


def linear_interpolate_batch_dt_trajectory_kernel(*args, **kwargs):
    raise NotImplementedError(
        "the raw Warp kernel is CUDA-only; call get_cuda_linear_interpolation instead"
    )


__all__ = ["get_cuda_linear_interpolation", "linear_interpolate_batch_dt_trajectory_kernel"]
