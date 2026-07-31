"""Shared portable helpers for the cuRobo mapper compatibility namespace."""

from __future__ import annotations

from typing import Any

import torch


RAW_KERNEL_MESSAGE = (
    "This is a raw Warp/CUDA mapper kernel surface. curobo-metal provides the "
    "equivalent high-level Mapper operation on CPU/MPS, but no raw Warp kernel ABI."
)


def unsupported_kernel(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(RAW_KERNEL_MESSAGE)


def dense_state(value: Any):
    """Return the native dense mapper state from a supported wrapper."""
    if hasattr(value, "_mapper"):
        value = value._mapper
    if hasattr(value, "state"):
        return value.state
    if all(hasattr(value, name) for name in ("tsdf", "weight", "occupancy", "esdf")):
        return value
    raise TypeError("expected a portable Mapper, PerceptionMapper, or DenseMap")


def require_cpu_or_mps(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if tensor.device.type not in {"cpu", "mps"}:
        raise ValueError("portable perception supports CPU and MPS tensors")
    return tensor
