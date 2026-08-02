"""Import-safe boundary for cuRoboV2's raw Warp swept-collision kernel."""

from __future__ import annotations

from typing import Any

# V2 compiles this value into the Warp kernel.  It is informational here;
# portable swept collision is sampled by ``SceneCollision`` instead.
SWEEP_STEPS = 3


def swept_sphere_obstacle_collision_kernel(
    obs_set: Any,
    spheres,
    weight,
    activation_distance,
    env_query_idx,
    distance,
    gradient,
    batch_size,
    horizon,
    num_spheres,
    max_n_obs,
    use_multi_env,
) -> None:
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
