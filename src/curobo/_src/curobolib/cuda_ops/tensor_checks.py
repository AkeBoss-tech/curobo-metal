from __future__ import annotations

import torch
from curobo._src.util.logging import log_and_raise


def _check_tensors(
    device: torch.device, expected_dtype: torch.dtype, **tensors: torch.Tensor
) -> None:
    """Validate the tensor preconditions shared by raw-kernel facades."""
    for name, tensor in tensors.items():
        if tensor is None:
            raise ValueError(f"{name}: expected a tensor, got None")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name}: expected a tensor, got {type(tensor).__name__}")
        if tensor.device != device:
            raise ValueError(f"{name}: expected device {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name}: expected contiguous tensor, got strides={tensor.stride()} "
                f"for shape={tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{name}: expected dtype {expected_dtype}, got {tensor.dtype}"
            )


def check_bool_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.bool, **tensors)


def check_float16_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.float16, **tensors)


def check_float32_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.float32, **tensors)


def check_int16_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.int16, **tensors)


def check_int32_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.int32, **tensors)


def check_int8_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.int8, **tensors)


def check_uint8_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    _check_tensors(device, torch.uint8, **tensors)
