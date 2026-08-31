"""Differentiable wxyz quaternion helpers implemented with PyTorch."""

from __future__ import annotations

from typing import Optional
import torch
from curobo._src.util.torch_util import get_torch_jit_decorator


def normalize_quaternion(in_quaternion: torch.Tensor) -> torch.Tensor:
    return in_quaternion / torch.linalg.vector_norm(in_quaternion, dim=-1, keepdim=True).clamp_min(
        torch.finfo(in_quaternion.dtype).eps
    )


def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, q_res: Optional[torch.Tensor] = None
) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    result = torch.stack((
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ), -1)
    if q_res is not None:
        q_res.copy_(result)
        return q_res
    return result


def angular_distance_phi3(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    goal = normalize_quaternion(goal_quat)
    current = normalize_quaternion(current_quat)
    dot = (goal * current).sum(-1).abs().clamp(min=0.0, max=1.0)
    return torch.acos(dot) / (torch.pi * 0.5)


def angular_distance_axis_angle(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    goal = normalize_quaternion(goal_quat)
    current = normalize_quaternion(current_quat)
    current_conjugate = current.clone()
    current_conjugate[..., 1:] *= -1.0
    relative = quat_multiply(goal, current_conjugate)
    vector_norm = torch.linalg.vector_norm(relative[..., 1:], dim=-1, keepdim=True)
    return 2.0 * torch.atan2(vector_norm, relative[..., 0].abs())


__all__ = [
    "angular_distance_axis_angle", "angular_distance_phi3",
    "normalize_quaternion", "quat_multiply",
]
