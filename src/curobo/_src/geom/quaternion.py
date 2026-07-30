"""Differentiable wxyz quaternion helpers implemented with PyTorch."""

from typing import Optional
import torch


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        torch.finfo(quaternion.dtype).eps
    )


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ), -1)


def angular_distance_phi3(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    goal = normalize_quaternion(goal_quat)
    current = normalize_quaternion(current_quat)
    return 1.0 - (goal * current).sum(-1).abs()


def angular_distance_axis_angle(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    goal = normalize_quaternion(goal_quat)
    current = normalize_quaternion(current_quat)
    dot = (goal * current).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


__all__ = [
    "angular_distance_axis_angle", "angular_distance_phi3",
    "normalize_quaternion", "quat_multiply",
]
