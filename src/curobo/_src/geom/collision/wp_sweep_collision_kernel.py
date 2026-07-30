"""Explicit boundary for CUDA/Warp swept collision kernels."""


def _unsupported(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "raw Warp swept kernels/analytic CCD are unavailable; use "
        "SceneCollision.get_swept_sphere_distance_raw for sampled swept queries"
    )


SWEEP_STEPS = 16


def swept_sphere_obstacle_collision_kernel(*args, **kwargs):
    return _unsupported(*args, **kwargs)


__all__ = ["SWEEP_STEPS", "swept_sphere_obstacle_collision_kernel"]
