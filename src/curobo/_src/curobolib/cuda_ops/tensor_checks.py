from __future__ import annotations

import torch


def _check_tensors(
    device: torch.device, expected_dtype: torch.dtype, **tensors: torch.Tensor
) -> None:
    """Validate the tensor preconditions shared by raw-kernel facades."""
    expected = torch.device(device)
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != expected:
            raise ValueError(f"{name} must be on {expected}, got {tensor.device}")
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"{name} must have dtype {expected_dtype}, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


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
