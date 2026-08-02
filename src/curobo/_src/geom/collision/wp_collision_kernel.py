"""Import-safe boundary for the pinned generic Warp collision kernel.

The argument layout deliberately mirrors cuRoboV2's Warp kernel.  Calling a
Warp kernel directly is not a portable operation: its first parameter is a
Warp/BVH obstacle descriptor and the remaining buffers are raw device arrays.
Use :class:`~curobo._src.geom.collision.collision_scene.SceneCollision` for the
CPU/MPS tensor implementation instead.
"""

from __future__ import annotations

from typing import Any


def sphere_obstacle_collision_kernel(
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
    """Pinned Warp launch layout; unavailable on the portable backend.

    This is intentionally not a tensor fallback.  In particular, accepting
    Warp descriptors and silently returning a tensor result would hide a
    CUDA/BVH dependency and make gradients or obstacle ties ambiguous.
    """
    del (
        obs_set, spheres, weight, activation_distance, env_query_idx, distance,
        gradient, batch_size, horizon, num_spheres, max_n_obs, use_multi_env,
    )
    raise NotImplementedError(
        "sphere_obstacle_collision_kernel requires Warp raw obstacle descriptors "
        "and CUDA device-array launches; use SceneCollision.get_sphere_distance "
        "or CollisionChecker.get_sphere_distance on CPU/MPS"
    )


__all__ = ["sphere_obstacle_collision_kernel"]
