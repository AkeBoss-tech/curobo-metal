"""Tensor-only helpers backing JointState operations."""

from __future__ import annotations

from typing import Tuple, Union

import torch

from curobo._src.util.tensor_util import clone_if_not_none, tensor_repeat_seeds
from curobo._src.util.torch_util import get_torch_jit_decorator


def jit_js_scale(
    vel: Union[None, torch.Tensor],
    acc: Union[None, torch.Tensor],
    jerk: Union[None, torch.Tensor],
    dt: torch.Tensor,
    new_dt: torch.Tensor,
):
    ratio = dt / new_dt
    return (
        None if vel is None else vel * ratio,
        None if acc is None else acc * ratio**2,
        None if jerk is None else jerk * ratio**3,
    )


def jit_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: torch.Tensor,
):
    return tuple(None if value is None else value[idx] for value in (position, velocity, acc, jerk, dt))


def fn_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: torch.Tensor,
):
    return tuple(None if value is None else value[idx] for value in (position, velocity, acc, jerk, dt))


def jit_get_index_int(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: int,
):
    return jit_get_index(position, velocity, acc, jerk, dt, idx)


def jit_inplace_reindex(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    knot: Union[torch.Tensor, None],
    new_index: torch.Tensor,
):
    values = (position, velocity, acceleration, jerk, knot)
    return tuple(None if value is None else torch.index_select(value, -1, new_index) for value in values)


def jit_joint_state_repeat_seeds(
    position: Union[torch.Tensor, None],
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    num_seeds: int,
):
    def repeat(value):
        if value is None:
            return None
        return value[:, None].repeat(1, num_seeds, *([1] * (value.ndim - 1))).reshape(
            value.shape[0] * num_seeds, *value.shape[1:]
        )
    return tuple(repeat(value) for value in (position, velocity, acceleration, jerk, dt))


def jit_joint_state_copy(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    in_position: torch.Tensor,
    in_velocity: Union[torch.Tensor, None],
    in_acceleration: Union[torch.Tensor, None],
    in_jerk: Union[torch.Tensor, None],
    in_dt: Union[torch.Tensor, None],
) -> Tuple[
    torch.Tensor,
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
]:
    for target, source in zip(
        (position, velocity, acceleration, jerk, dt),
        (in_position, in_velocity, in_acceleration, in_jerk, in_dt),
    ):
        if target is not None and source is not None:
            target.copy_(source)
    return position, velocity, acceleration, jerk, dt


def clone_state_jit(
    position: Union[torch.Tensor, None],
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
) -> Tuple[
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
]:
    return tuple(None if value is None else value.clone() for value in
                 (position, velocity, acceleration, jerk, dt))


def trim_trajectory_jit(position, velocity, acceleration, jerk, start_idx: int, end_idx: int):
    return tuple(None if value is None else value[..., start_idx:end_idx, :] for value in
                 (position, velocity, acceleration, jerk))
