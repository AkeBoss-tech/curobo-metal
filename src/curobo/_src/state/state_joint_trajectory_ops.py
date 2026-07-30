"""Batched trajectory indexing operations."""

import torch


def gather_joint_state_by_seed(joint_state, idx):
    if idx.ndim != 2 or joint_state.position.ndim != 4:
        raise ValueError("expected idx [batch,topk] and state [batch,seeds,horizon,dof]")
    batch = torch.arange(idx.shape[0], device=idx.device)[:, None]
    return joint_state._shape_map(lambda value: value[batch, idx])


def copy_joint_state_only_index(target, source, idx):
    target[idx] = source[idx]
    return target


def copy_joint_state_at_index(target, source, idx):
    target[idx] = source
    return target


def copy_joint_state_at_batch_seed_indices(target, source, batch_idx, seed_idx):
    for name in ("position", "velocity", "acceleration", "jerk", "knot", "knot_dt", "dt"):
        a, b = getattr(target, name), getattr(source, name)
        if a is not None and b is not None:
            a[batch_idx, seed_idx] = b[batch_idx, seed_idx]
    return target


def get_joint_state_at_horizon_index(joint_state, horizon_index: int):
    return joint_state._shape_map(lambda value: value[..., horizon_index, :])


def trim_joint_state_trajectory(joint_state, start_idx: int, end_idx=None):
    end_idx = joint_state.position.shape[-2] if end_idx in (None, 0) else end_idx
    return joint_state._shape_map(lambda value: value[..., start_idx:end_idx, :])


def index_joint_state_dof(joint_state, idx):
    from .state_joint import JointState
    def select(value):
        return None if value is None else torch.index_select(value, -1, idx)
    names = None
    if joint_state.joint_names is not None:
        names = [joint_state.joint_names[int(i)] for i in idx]
    return JointState(
        select(joint_state.position), select(joint_state.velocity),
        select(joint_state.acceleration), names, select(joint_state.jerk),
        dt=joint_state.dt, knot=select(joint_state.knot),
        knot_dt=joint_state.knot_dt, control_space=joint_state.control_space,
    )
