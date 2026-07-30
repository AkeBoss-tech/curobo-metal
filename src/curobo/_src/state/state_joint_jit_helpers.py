"""Tensor-only helpers backing JointState operations."""

import torch


def jit_js_scale(vel, acc, jerk, dt, new_dt):
    ratio = dt / new_dt
    return (
        None if vel is None else vel * ratio,
        None if acc is None else acc * ratio**2,
        None if jerk is None else jerk * ratio**3,
    )


def jit_get_index(position, velocity, acc, jerk, dt, idx):
    return tuple(None if value is None else value[idx] for value in (position, velocity, acc, jerk, dt))


fn_get_index = jit_get_index


def jit_get_index_int(position, velocity, acc, jerk, dt, idx: int):
    return jit_get_index(position, velocity, acc, jerk, dt, idx)


def jit_inplace_reindex(position, velocity, acceleration, jerk, knot, new_index):
    values = (position, velocity, acceleration, jerk, knot)
    return tuple(None if value is None else torch.index_select(value, -1, new_index) for value in values)


def jit_joint_state_repeat_seeds(position, velocity, acceleration, jerk, dt, num_seeds: int):
    def repeat(value):
        if value is None:
            return None
        return value[:, None].repeat(1, num_seeds, *([1] * (value.ndim - 1))).reshape(
            value.shape[0] * num_seeds, *value.shape[1:]
        )
    return tuple(repeat(value) for value in (position, velocity, acceleration, jerk, dt))


def jit_joint_state_copy(position, velocity, acceleration, jerk, dt,
                         in_position, in_velocity, in_acceleration, in_jerk, in_dt):
    for target, source in zip(
        (position, velocity, acceleration, jerk, dt),
        (in_position, in_velocity, in_acceleration, in_jerk, in_dt),
    ):
        if target is not None and source is not None:
            target.copy_(source)
    return position, velocity, acceleration, jerk, dt


def clone_state_jit(position, velocity, acceleration, jerk, dt):
    return tuple(None if value is None else value.clone() for value in
                 (position, velocity, acceleration, jerk, dt))


def trim_trajectory_jit(position, velocity, acceleration, jerk, start_idx: int, end_idx: int):
    return tuple(None if value is None else value[..., start_idx:end_idx, :] for value in
                 (position, velocity, acceleration, jerk))
