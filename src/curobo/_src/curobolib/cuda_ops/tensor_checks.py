from __future__ import annotations

import torch


def _check(device, dtype, tensors):
    expected = torch.device(device)
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != expected:
            raise ValueError(f"{name} must be on {expected}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def check_bool_tensors(device, tensors=None, **kwargs): _check(device, torch.bool, kwargs if tensors is None else tensors)
def check_float16_tensors(device, tensors=None, **kwargs): _check(device, torch.float16, kwargs if tensors is None else tensors)
def check_float32_tensors(device, tensors=None, **kwargs): _check(device, torch.float32, kwargs if tensors is None else tensors)
def check_int16_tensors(device, tensors=None, **kwargs): _check(device, torch.int16, kwargs if tensors is None else tensors)
def check_int32_tensors(device, tensors=None, **kwargs): _check(device, torch.int32, kwargs if tensors is None else tensors)
def check_int8_tensors(device, tensors=None, **kwargs): _check(device, torch.int8, kwargs if tensors is None else tensors)
def check_uint8_tensors(device, tensors=None, **kwargs): _check(device, torch.uint8, kwargs if tensors is None else tensors)
