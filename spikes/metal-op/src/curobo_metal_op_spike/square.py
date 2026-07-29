"""Autograd-compatible square implemented by custom Metal compute kernels."""

from __future__ import annotations

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void square_forward(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = input[index] * input[index];
}

kernel void square_backward(
    const device float* input [[buffer(0)]],
    const device float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  grad_input[index] = 2.0f * input[index] * grad_output[index];
}
"""

_library = None


def _metal_library():
    global _library
    if _library is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("torch.mps.compile_shader requires PyTorch 2.13 or newer")
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def _validate(input: torch.Tensor) -> None:
    if input.device.type != "mps":
        raise ValueError("metal_square requires an MPS tensor; CPU fallback is forbidden")
    if input.dtype != torch.float32:
        raise TypeError("metal_square currently supports only torch.float32")
    if not input.is_contiguous():
        raise ValueError("metal_square currently requires a contiguous tensor")


class _MetalSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        _validate(input)
        output = torch.empty_like(input)
        if input.numel():
            _metal_library().square_forward(input, output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (input,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(input)
        if input.numel():
            _metal_library().square_backward(input, grad_output, grad_input)
        return (grad_input,)


def metal_square(input: torch.Tensor) -> torch.Tensor:
    """Return ``input**2`` using custom forward and backward Metal kernels."""
    return _MetalSquare.apply(input)

