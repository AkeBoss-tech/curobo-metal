"""Import-safe boundary for the pinned generic Warp collision kernel.

The argument layout deliberately mirrors cuRoboV2's Warp kernel.  Calling a
Warp kernel directly is not a portable operation: its first parameter is a
Warp/BVH obstacle descriptor and the remaining buffers are raw device arrays.
Use :class:`~curobo._src.geom.collision.collision_scene.SceneCollision` for the
CPU/MPS tensor implementation instead.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

wp = None
OBSTACLE_SDF_MODULES = {}


def _raw_warp_unavailable(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("Warp/CUDA raw collision kernels are unavailable on Metal")


accumulate_collision = apply_collision_activation = compute_local_sdf_with_grad = is_obs_enabled = load_obstacle_transform = load_sphere_query = _raw_warp_unavailable


def sphere_obstacle_collision_kernel(
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
