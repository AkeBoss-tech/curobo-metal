"""Explicit boundaries for pinned Warp autograd launch wrappers.

The high-level :mod:`collision_scene` and :mod:`checker_collision` APIs are
fully tensor backed.  These classes intentionally retain the upstream callable
layout so callers fail at the right boundary instead of receiving fabricated
results from an incompatible raw CUDA launch.
"""

from __future__ import annotations

import torch

from .wp_collision_kernel import sphere_obstacle_collision_kernel
from .wp_speed_metric import apply_speed_metric
from .wp_sweep_collision_kernel import swept_sphere_obstacle_collision_kernel


def _raw_warp_error(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"{name} requires Warp CUDA kernel launch buffers; use the "
        "differentiable SceneCollision or CollisionChecker tensor API on CPU/MPS"
    )


class SphereObstacleCollision(torch.autograd.Function):
    @classmethod
    def apply(cls, *args, **kwargs):
        """Fail before PyTorch binds raw launch arguments.

        The inherited ``Function.apply`` otherwise raises a misleading Python
        argument error for an accidental zero-argument call instead of the
        explicit Warp/CUDA boundary this compatibility layer promises.
        """
        del args, kwargs
        raise _raw_warp_error("SphereObstacleCollision")

    @staticmethod
    def forward(
        ctx,
        query_spheres: torch.Tensor,
        buffer,
        scene,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        max_distance: torch.Tensor,
        env_query_idx: torch.Tensor,
        use_multi_env: bool,
        return_loss: bool = False,
    ) -> torch.Tensor:
        del (
            ctx, query_spheres, buffer, scene, weight, activation_distance,
            max_distance, env_query_idx, use_multi_env, return_loss,
        )
        raise _raw_warp_error("SphereObstacleCollision")

    @staticmethod
    def backward(ctx, grad_output):
        del ctx, grad_output
        raise _raw_warp_error("SphereObstacleCollision.backward")


class SweptSphereObstacleCollision(torch.autograd.Function):
    @classmethod
    def apply(cls, *args, **kwargs):
        del args, kwargs
        raise _raw_warp_error("SweptSphereObstacleCollision")

    @staticmethod
    def forward(
        ctx,
        query_spheres: torch.Tensor,
        buffer,
        scene,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        max_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        enable_speed_metric: bool,
        env_query_idx: torch.Tensor,
        use_multi_env: bool,
        return_loss: bool = False,
    ) -> torch.Tensor:
        del (
            ctx, query_spheres, buffer, scene, weight, activation_distance,
            max_distance, speed_dt, enable_speed_metric, env_query_idx,
            use_multi_env, return_loss,
        )
        raise _raw_warp_error("SweptSphereObstacleCollision")

    @staticmethod
    def backward(ctx, grad_output):
        del ctx, grad_output
        raise _raw_warp_error("SweptSphereObstacleCollision.backward")


__all__ = [
    "SphereObstacleCollision", "SweptSphereObstacleCollision", "apply_speed_metric",
    "sphere_obstacle_collision_kernel", "swept_sphere_obstacle_collision_kernel",
]
