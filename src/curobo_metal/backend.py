"""Device policy and backend dispatch for production cuRobo Metal operators."""

from __future__ import annotations

from enum import Enum

import torch


class Backend(str, Enum):
    """Backends implemented by this package."""

    CPU = "cpu"
    MPS = "mps"


def default_device() -> torch.device:
    """Select the available native accelerator for an omitted device request.

    Explicit requests are validated separately; availability never redirects an
    explicitly requested MPS device to CPU.
    """
    # Allocated MPS tensors report index 0; match their device exactly so public
    # configuration/tensor equality checks work just as with upstream cuda:0.
    return torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve and validate a supported, available compute device."""
    resolved = default_device() if device is None else torch.device(device)
    if resolved.type not in {backend.value for backend in Backend}:
        raise ValueError(
            f"unsupported device {resolved}; curobo-metal supports only cpu and mps"
        )
    if resolved.index not in (None, 0):
        raise ValueError(f"device indices are not supported: {resolved}")
    if resolved.type == Backend.MPS.value and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("mps:0" if resolved.type == Backend.MPS.value else "cpu")


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
