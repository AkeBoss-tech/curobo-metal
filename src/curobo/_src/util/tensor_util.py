"""Differentiable tensor utilities for CPU and MPS."""

from __future__ import annotations

from typing import Protocol
import torch


class TensorLike(Protocol):
    @property
    def shape(self): ...
    def copy_(self, other): ...
    def clone(self): ...


def check_tensor_shapes(new_tensor, mem_tensor):
    return isinstance(mem_tensor, torch.Tensor) and new_tensor.shape == mem_tensor.shape


def copy_tensor(new_tensor, mem_tensor):
    if check_tensor_shapes(new_tensor, mem_tensor):
        mem_tensor.copy_(new_tensor)
        return True
    return False


def copy_or_clone(new_tensor, ref_tensor, allow_clone: bool = True):
    if new_tensor is None:
        return ref_tensor
    if ref_tensor is None:
        if not allow_clone:
            raise ValueError("ref_tensor is None")
        return new_tensor.clone()
    if ref_tensor.shape != new_tensor.shape:
        if allow_clone:
            return new_tensor.clone()
        raise ValueError(f"ref_tensor.shape {ref_tensor.shape} != new_tensor.shape {new_tensor.shape}")
    ref_tensor.copy_(new_tensor)
    return ref_tensor


def clone_if_not_none(x):
    return None if x is None else x.clone()


def cat_sum(tensor_list, sum_dim=(0,)):
    return torch.cat(tensor_list, dim=-1).sum(dim=sum_dim)


def cat_max(tensor_list):
    return torch.stack(tensor_list).max(dim=0).values


def tensor_repeat_seeds(tensor, num_seeds: int):
    return tensor[:, None].repeat(1, num_seeds, *([1] * (tensor.ndim - 1))).reshape(
        tensor.shape[0] * num_seeds, *tensor.shape[1:]
    )


def fd_tensor(p, dt):
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


def check_nan_last_dimension(position_trajectory):
    return torch.isnan(position_trajectory).any(dim=-1)


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


def round_away_from_zero(x):
    return torch.trunc(x + 0.5 * torch.sign(x))


def stable_topk(input_tensor, k: int, dim: int = -1, largest: bool = True):
    return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=True)
