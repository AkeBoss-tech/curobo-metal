"""Backend-neutral collision activation helpers.

Raw Warp array descriptors are outside the Metal port; tensor callers should
use :class:`SceneCollision` and :class:`CollisionBuffer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SphereQueryData:
    query_spheres: torch.Tensor
    weight: torch.Tensor
    activation_distance: torch.Tensor


def apply_collision_activation(distance: torch.Tensor, activation_distance) -> torch.Tensor:
    activation = torch.as_tensor(
        activation_distance, device=distance.device, dtype=distance.dtype
    )
    return distance - activation


def load_sphere_query(query_spheres, weight=1.0, activation_distance=0.0):
    if not isinstance(query_spheres, torch.Tensor):
        raise TypeError("the portable backend requires torch.Tensor query spheres")
    return SphereQueryData(
        query_spheres,
        torch.as_tensor(weight, device=query_spheres.device, dtype=query_spheres.dtype),
        torch.as_tensor(
            activation_distance, device=query_spheres.device, dtype=query_spheres.dtype
        ),
    )

def accumulate_collision(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "Warp atomic collision accumulation is unavailable; SceneCollision "
        "provides deterministic tensor reduction"
    )


def process_collision_result(distance: torch.Tensor, weight=1.0):
    return distance * torch.as_tensor(weight, device=distance.device, dtype=distance.dtype)


__all__ = [
    "SphereQueryData", "accumulate_collision", "apply_collision_activation",
    "load_sphere_query", "process_collision_result",
]
