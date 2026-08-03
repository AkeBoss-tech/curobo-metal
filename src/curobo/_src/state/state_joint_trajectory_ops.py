"""Portable batched JointState trajectory operations.

These use native PyTorch indexing and assignment so CPU and Apple Metal share
the same semantics.  They intentionally do not expose cuRobo's CUDA JIT
buffer ABI.
"""

from __future__ import annotations

import torch


_FIELDS = ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt")
_UNSET = object()


def _clone(value):
    return None if value is None else value.clone()


def _state_like(state, *, joint_names=_UNSET, **changes):
    from .state_joint import JointState

    values = {name: _clone(getattr(state, name)) for name in _FIELDS}
    values.update(changes)
    return JointState(
        position=values["position"], velocity=values["velocity"],
        acceleration=values["acceleration"], jerk=values["jerk"],
        joint_names=(None if state.joint_names is None else state.joint_names.copy())
        if joint_names is _UNSET else joint_names,
        device_cfg=state.device_cfg, dt=values["dt"], knot=values["knot"],
        knot_dt=values["knot_dt"], aux_data=dict(state.aux_data),
        control_space=state.control_space,
    )


def gather_joint_state_by_seed(joint_state, idx: torch.Tensor):
    """Gather ``[batch, topk]`` seed choices from a ``[batch, seed, ...]`` state."""
    if not isinstance(idx, torch.Tensor) or idx.ndim != 2:
        raise ValueError("idx must be a rank-two tensor [batch, topk]")
    if idx.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("idx must have an integer dtype")
    if joint_state.position.ndim != 4:
        raise ValueError("position must have shape [batch, seeds, horizon, dof]")
    batch_size, seeds = joint_state.position.shape[:2]
    if idx.shape[0] != batch_size:
        raise ValueError("idx batch size does not match position batch size")
    index = idx.to(device=joint_state.device, dtype=torch.long)
    batch = torch.arange(batch_size, device=joint_state.device)[:, None]

    def gather(value):
        if value is None:
            return None
        if value.ndim < 2 or value.shape[:2] != (batch_size, seeds):
            # Global timing metadata is shared, not seed-indexed.
            return value.clone()
        return value[batch, index]

    return _state_like(joint_state, **{name: gather(getattr(joint_state, name)) for name in _FIELDS})


def copy_joint_state_only_index(target, source, idx):
    """Copy matching indexed entries from source to target, in place."""
    with torch.no_grad():
        for name in _FIELDS:
            dst, src = getattr(target, name), getattr(source, name)
            if dst is not None and src is not None:
                dst[idx] = src[idx]
    return target


def copy_joint_state_at_index(target, source, idx):
    """Copy an unbatched source state into one or more target batch slots."""
    if isinstance(idx, int):
        maximum = idx
    elif isinstance(idx, torch.Tensor):
        if idx.numel() == 0:
            return target
        maximum = int(idx.max().item())
    else:
        if not idx:
            return target
        maximum = max(idx)
    if maximum >= target.position.shape[0] or maximum < -target.position.shape[0]:
        raise ValueError(f"{maximum} index out of range, current state is of length {target.position.shape[0]}")
    with torch.no_grad():
        for name in _FIELDS:
            dst, src = getattr(target, name), getattr(source, name)
            if dst is not None and src is not None:
                dst[idx] = src
    return target


def copy_joint_state_at_batch_seed_indices(target, source, batch_idx, seed_idx):
    """Copy selected ``[batch, seed]`` entries through all materialized fields."""
    if not isinstance(batch_idx, torch.Tensor) or not isinstance(seed_idx, torch.Tensor):
        raise ValueError("batch_idx and seed_idx must be tensors")
    if batch_idx.shape != seed_idx.shape:
        raise ValueError("batch_idx and seed_idx must have equal shape")
    batch = batch_idx.to(device=target.device, dtype=torch.long)
    seed = seed_idx.to(device=target.device, dtype=torch.long)
    with torch.no_grad():
        for name in _FIELDS:
            dst, src = getattr(target, name), getattr(source, name)
            if dst is not None and src is not None:
                if dst.ndim >= 2 and src.ndim >= 2:
                    dst[batch, seed] = src[batch, seed]
    return target


def get_joint_state_at_horizon_index(joint_state, horizon_index: int):
    """Select a waypoint along the final non-DOF (horizon) axis."""
    if joint_state.position.ndim < 2:
        raise ValueError("JointState does not have horizon")

    def select(value):
        return None if value is None else value[..., horizon_index, :]

    knot = joint_state.knot
    if knot is not None:
        # A spline may intentionally have fewer knots than the sampled dense
        # horizon.  Select it only when the requested dense waypoint is also
        # a valid knot; otherwise retain the parameterization unchanged.
        knot = select(knot) if -knot.shape[-2] <= horizon_index < knot.shape[-2] else knot.clone()
    return _state_like(
        joint_state,
        position=select(joint_state.position), velocity=select(joint_state.velocity),
        acceleration=select(joint_state.acceleration), jerk=select(joint_state.jerk), knot=knot,
    )


def trim_joint_state_trajectory(joint_state, start_idx: int, end_idx=None):
    """Return a trajectory slice without retaining stale knot parameterization."""
    if joint_state.position.ndim < 2:
        raise ValueError("JointState does not have horizon")
    horizon = joint_state.position.shape[-2]
    stop = horizon if end_idx in (None, 0) else int(end_idx)
    if start_idx < -horizon or start_idx > horizon or stop < -horizon or stop > horizon:
        raise ValueError("trajectory trim index out of range")

    def trim(value):
        return None if value is None else value[..., start_idx:stop, :]

    # This mirrors V2: a sliced dense trajectory no longer carries an
    # unmodified spline knot parameterization whose time origin is stale.
    return _state_like(
        joint_state,
        position=trim(joint_state.position), velocity=trim(joint_state.velocity),
        acceleration=trim(joint_state.acceleration), jerk=trim(joint_state.jerk),
        knot=None, knot_dt=None,
    )


def index_joint_state_dof(joint_state, idx):
    """Select joint/DOF columns while retaining timing and solver metadata."""
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx, device=joint_state.device)
    index = idx.to(device=joint_state.device, dtype=torch.long).reshape(-1)
    if index.numel() == 0:
        raise ValueError("idx must select at least one DOF")
    dof = joint_state.position.shape[-1]
    if bool(((index < -dof) | (index >= dof)).any().item()):
        raise ValueError("DOF index out of range")
    index = torch.remainder(index, dof)

    def select(value):
        return None if value is None else torch.index_select(value, -1, index)

    names = None
    if joint_state.joint_names is not None:
        names = [joint_state.joint_names[int(i)] for i in index.cpu().tolist()]
    result = _state_like(
        joint_state,
        joint_names=None,
        position=select(joint_state.position), velocity=select(joint_state.velocity),
        acceleration=select(joint_state.acceleration), jerk=select(joint_state.jerk), knot=select(joint_state.knot),
    )
    result.joint_names = names
    return result
