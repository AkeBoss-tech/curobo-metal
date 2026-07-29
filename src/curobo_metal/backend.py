"""Device policy and backend dispatch for production cuRobo Metal operators."""

from __future__ import annotations

from enum import Enum

import torch


class Backend(str, Enum):
    """Backends implemented by this package."""

    CPU = "cpu"
    MPS = "mps"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve and validate a supported, available compute device."""
    resolved = torch.device("cpu" if device is None else device)
    if resolved.type not in {backend.value for backend in Backend}:
        raise ValueError(
            f"unsupported device {resolved}; curobo-metal supports only cpu and mps"
        )
    if resolved.index not in (None, 0):
        raise ValueError(f"device indices are not supported: {resolved}")
    if resolved.type == Backend.MPS.value and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(resolved.type)


def validate_tensor_device(
    tensor: torch.Tensor,
    *,
    expected: str | torch.device | None = None,
) -> torch.device:
    """Reject unsupported or unexpected devices instead of silently copying."""
    actual = resolve_device(tensor.device)
    if expected is not None:
        wanted = resolve_device(expected)
        if actual != wanted:
            raise ValueError(f"expected a tensor on {wanted}, got {tensor.device}")
    return actual


def synchronize(device: str | torch.device) -> None:
    """Synchronize asynchronous work for timing or host observation."""
    resolved = resolve_device(device)
    if resolved.type == Backend.MPS.value:
        torch.mps.synchronize()
