"""Import-safe boundary for cuRoboV2's raw Warp swept-collision kernel."""

from __future__ import annotations

from importlib import import_module
from typing import Any

wp = None
OBSTACLE_SDF_MODULES = {}


def _raw_warp_unavailable(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("Warp/CUDA raw collision kernels are unavailable on Metal")


accumulate_collision = apply_collision_activation = compute_local_sdf = compute_local_sdf_with_grad = is_obs_enabled = load_obstacle_transform = load_sphere_query = _raw_warp_unavailable

# V2 compiles this value into the Warp kernel.  It is informational here;
# portable swept collision is sampled by ``SceneCollision`` instead.
SWEEP_STEPS = 3


def swept_sphere_obstacle_collision_kernel(
    obs_set: Any,
    spheres: wp.array(dtype=wp.vec4),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
):
    """Pinned raw Warp launch layout; analytic Warp CCD is unsupported."""
    del (
        obs_set, spheres, weight, activation_distance, env_query_idx, distance,
        gradient, batch_size, horizon, num_spheres, max_n_obs, use_multi_env,
    )
    raise NotImplementedError(
        "swept_sphere_obstacle_collision_kernel requires Warp/CUDA raw buffers; "
        "use SceneCollision.get_swept_sphere_distance for the documented "
        "fixed-resolution sampled CPU/MPS query"
    )


__all__ = ["SWEEP_STEPS", "swept_sphere_obstacle_collision_kernel"]
