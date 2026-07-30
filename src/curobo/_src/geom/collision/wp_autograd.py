"""Compatibility boundary for upstream Warp autograd launch wrappers."""

from .wp_collision_kernel import sphere_obstacle_collision_kernel
from .wp_speed_metric import apply_speed_metric
from .wp_sweep_collision_kernel import swept_sphere_obstacle_collision_kernel


class _WarpOnly:
    @staticmethod
    def apply(*args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "raw Warp kernel launches are unavailable; use SceneCollision's "
            "differentiable PyTorch discrete or swept query APIs"
        )


class SphereObstacleCollision(_WarpOnly):
    pass


class SweptSphereObstacleCollision(_WarpOnly):
    pass


__all__ = [
    "SphereObstacleCollision", "SweptSphereObstacleCollision",
    "apply_speed_metric", "sphere_obstacle_collision_kernel",
    "swept_sphere_obstacle_collision_kernel",
]
