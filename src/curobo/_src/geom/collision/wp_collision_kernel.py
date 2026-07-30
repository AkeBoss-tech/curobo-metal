"""Explicit boundary for CUDA/Warp collision kernels."""


def _unsupported(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "raw Warp collision kernels are unavailable; use SceneCollision"
    )


def sphere_obstacle_collision_kernel(*args, **kwargs):
    return _unsupported(*args, **kwargs)

__all__ = ["sphere_obstacle_collision_kernel"]
