"""Differentiable tensor utilities for CPU and MPS."""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, Union
import torch

from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class TensorLike(Protocol):
    @property
    def shape(self): ...
    def copy_(self, other): ...
    def clone(self): ...


def check_tensor_shapes(new_tensor: torch.Tensor, mem_tensor: torch.Tensor):
    return (
        isinstance(mem_tensor, torch.Tensor)
        and len(mem_tensor.shape) == len(new_tensor.shape)
        and new_tensor.shape == mem_tensor.shape
    )


def copy_tensor(new_tensor: torch.Tensor, mem_tensor: torch.Tensor):
    if check_tensor_shapes(new_tensor, mem_tensor):
        mem_tensor.copy_(new_tensor)
        return True
    return False


def copy_or_clone(
    new_tensor: Optional[TensorLike],
    ref_tensor: Optional[TensorLike],
    allow_clone: bool = True,
) -> Optional[TensorLike]:
    """Copy into a compatible buffer or return a new portable buffer.

    The ``None`` behaviour matches cuRobo's caller-owned-buffer convention:
    a missing input preserves the reference, and a missing reference clones
    the input only when cloning is enabled.
    """
    if ref_tensor is None and new_tensor is None:
        return None
    if new_tensor is None:
        return ref_tensor
    if ref_tensor is None:
        if allow_clone:
            return new_tensor.clone()
        log_and_raise("ref_tensor is None")
    if ref_tensor.shape == new_tensor.shape:
        ref_tensor.copy_(new_tensor)
        return ref_tensor
    if allow_clone:
        return new_tensor.clone()
    log_and_raise(f"ref_tensor.shape {ref_tensor.shape} != new_tensor.shape {new_tensor.shape}")


def clone_if_not_none(x: Union[torch.Tensor, None]) -> Union[torch.Tensor, None]:
    return None if x is None else x.clone()


@get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
def cat_sum(
    tensor_list: List[torch.Tensor], sum_dim: Union[Tuple[int, int], Tuple[int], int] = (0,)
):
    return torch.cat(tensor_list, dim=-1).sum(dim=sum_dim)


@get_torch_jit_decorator(slow_to_compile=True)
def cat_max(tensor_list: List[torch.Tensor]):
    return torch.stack(tensor_list).max(dim=0).values


@get_torch_jit_decorator(slow_to_compile=True)
def tensor_repeat_seeds(tensor: torch.Tensor, num_seeds: int):
    return tensor[:, None].repeat(1, num_seeds, *([1] * (tensor.ndim - 1))).reshape(
        tensor.shape[0] * num_seeds, *tensor.shape[1:]
    )


@get_torch_jit_decorator(slow_to_compile=True)
def fd_tensor(p: torch.Tensor, dt: torch.Tensor):
    delta = p[..., 1:, :] - p[..., :-1, :]
    step = torch.as_tensor(dt, device=p.device, dtype=p.dtype)
    if step.ndim == 0:
        return delta / step
    step = step[..., : delta.shape[-2]]
    if step.ndim == 1:
        step = step.view(*([1] * (delta.ndim - 2)), step.shape[0], 1)
    else:
        step = step.unsqueeze(-1)
    return delta / step


@get_torch_jit_decorator(slow_to_compile=True)
def check_nan_last_dimension(position_trajectory):
    return torch.isnan(position_trajectory).any(dim=-1)


@get_torch_jit_decorator(slow_to_compile=True)
def shift_buffer(buffer, shift_d: int, action_dim: int, shift_steps: int = 1):
    output = buffer.clone().roll(-shift_d, -2)
    end = -(shift_steps - 1) * action_dim or output.shape[-2]
    output[..., -shift_d:end, :] = output[..., -shift_d-action_dim:-shift_d, :].clone()
    return output


def jit_copy_buffer(ref_buffer, buffer):
    if buffer is None:
        return ref_buffer
    if ref_buffer is None:
        return buffer.clone()
    ref_buffer.copy_(buffer)
    return ref_buffer


def find_first_idx(array, value, EQUAL=False):
    return torch.nonzero(array >= value if EQUAL else array > value)[0].item()


def find_last_idx(array, value):
    return torch.nonzero(array <= value)[-1].item()


@get_torch_jit_decorator(slow_to_compile=True)
def round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.trunc(x + 0.5 * torch.sign(x))


def stable_topk(input_tensor: torch.Tensor, k: int, dim: int = -1, largest: bool = True):
    return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=True)
