"""Differentiable, portable operations on :class:`JointState` values.

The pinned CUDA implementation routes several of these through TorchScript
helpers.  These implementations deliberately use ordinary PyTorch tensor
operations instead: they work on CPU and MPS, keep autograd graphs intact,
and make no claim to reproduce CUDA packed-buffer/JIT ABI details.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from curobo._src.util.tensor_util import fd_tensor

from .filter_coeff import FilterCoeff


_CHANNELS = ("position", "velocity", "acceleration", "jerk")
_TENSOR_FIELDS = _CHANNELS + ("dt", "knot", "knot_dt")
_UNSET = object()


def _clone(value):
    return None if value is None else value.clone()


def _new_state(state, *, joint_names=_UNSET, **overrides):
    """Build a state without accidentally dropping trajectory metadata."""
    from .state_joint import JointState

    values = {name: _clone(getattr(state, name)) for name in _TENSOR_FIELDS}
    values.update(overrides)
    return JointState(
        position=values["position"], velocity=values["velocity"],
        acceleration=values["acceleration"], jerk=values["jerk"],
        joint_names=(None if state.joint_names is None else state.joint_names.copy())
        if joint_names is _UNSET else joint_names,
        device_cfg=state.device_cfg, dt=values["dt"], knot=values["knot"],
        knot_dt=values["knot_dt"], aux_data=dict(state.aux_data),
        control_space=state.control_space,
    )


def blend_joint_states(target, new_state, coeff: FilterCoeff):
    """Blend materialized channels into ``target`` in place.

    Partial states are valid in the portable facade.  A missing channel is
    therefore left untouched rather than synthesizing a hidden zero buffer.
    """
    for name in _CHANNELS:
        old, new = getattr(target, name), getattr(new_state, name)
        if old is None or new is None:
            continue
        if old.shape != new.shape or old.device != new.device:
            raise ValueError(f"{name} channels must share shape and device")
        weight = torch.as_tensor(getattr(coeff, name), dtype=old.dtype, device=old.device)
        old.copy_(weight * new + (1.0 - weight) * old)
    return target


def joint_state_to_tensor(joint_state):
    """Pack position plus the three derivative channels into ``[..., 4*dof]``."""
    zero = torch.zeros_like(joint_state.position)
    return torch.cat(
        tuple(getattr(joint_state, name) if getattr(joint_state, name) is not None else zero
              for name in _CHANNELS),
        dim=-1,
    )


def stack_joint_states(js1, js2):
    """Stack compatible states along the trajectory/sample axis."""
    if (
        js1.position.ndim < 2
        or js1.position.ndim != js2.position.ndim
        or js1.position.shape[:-2] != js2.position.shape[:-2]
        or js1.position.shape[-1] != js2.position.shape[-1]
    ):
        raise ValueError("stacked JointStates must share batch dimensions and DOF")
    if js1.position.device != js2.position.device:
        raise ValueError("stacked JointStates must be on the same device")
    # Construct via the canonical packed form to guarantee all four channels
    # have the same shape, including when either input is partial.
    from .state_joint import JointState
    packed = torch.cat((joint_state_to_tensor(js1), joint_state_to_tensor(js2)), dim=-2)
    result = JointState.from_state_tensor(packed, js1.joint_names, js1.position.shape[-1])
    result.dt = _clone(js1.dt)
    result.knot = _clone(js1.knot)
    result.knot_dt = _clone(js1.knot_dt)
    result.aux_data = dict(js1.aux_data)
    result.control_space = js1.control_space
    return result


def cat_joint_states(js1, js2, dim: int):
    """Concatenate state channels; DOF concatenation also joins joint names."""
    if js1.position.device != js2.position.device:
        raise ValueError("concatenated JointStates must be on the same device")
    if js1.position.ndim != js2.position.ndim:
        raise ValueError("concatenated JointStates must have equal rank")

    def cat(name):
        left, right = getattr(js1, name), getattr(js2, name)
        return None if left is None or right is None else torch.cat((left, right), dim=dim)

    dof_dim = dim if dim >= 0 else js1.position.ndim + dim
    if dof_dim == js1.position.ndim - 1:
        if js1.joint_names is None or js2.joint_names is None:
            names = None
        else:
            names = js1.joint_names + js2.joint_names
    else:
        names = None if js1.joint_names is None else js1.joint_names.copy()
    # The constructor checks joint-name cardinality, so defer attaching the
    # concatenated name list until after its final DOF width is known.
    result = _new_state(js1, joint_names=None, **{name: cat(name) for name in _CHANNELS})
    result.joint_names = names
    return result


def _repeat_metadata(value, repeats: tuple[int, ...]):
    if value is None:
        return None
    if value.ndim == 0:
        return value.clone()
    # Metadata represents a prefix of the position layout (never its DOF
    # axis), so use the corresponding prefix of the caller's repeat shape.
    if value.ndim > len(repeats):
        return value.clone()
    return value.repeat(*repeats[: value.ndim])


def repeat_joint_state(joint_state, repeat_input: Sequence[int]):
    """Repeat channels and batch-shaped timing/knot metadata."""
    repeats = tuple(int(value) for value in repeat_input)
    if len(repeats) != joint_state.position.ndim or any(value < 0 for value in repeats):
        raise ValueError("repeat_input must contain one non-negative value per position dimension")
    result = _new_state(
        joint_state,
        **{name: None if getattr(joint_state, name) is None else getattr(joint_state, name).repeat(*repeats)
           for name in _CHANNELS},
        dt=_repeat_metadata(joint_state.dt, repeats),
        knot=_repeat_metadata(joint_state.knot, repeats),
        knot_dt=_repeat_metadata(joint_state.knot_dt, repeats),
    )
    return result


def repeat_joint_state_seeds(joint_state, num_seeds: int):
    """Expand the first (batch) axis into contiguous batch-major seed rows."""
    if not isinstance(num_seeds, int) or num_seeds < 1:
        raise ValueError("num_seeds must be a positive integer")
    return joint_state.repeat_seeds(num_seeds)


def _apply_kernel(kernel_mat: torch.Tensor, value):
    if value is None:
        return None
    if kernel_mat.device != value.device:
        raise ValueError("kernel and JointState must be on the same device")
    if kernel_mat.shape[-1] != value.shape[0]:
        raise ValueError("kernel trailing dimension must match state leading dimension")
    # ``matmul`` treats the final two dimensions of a high-rank state as the
    # matrix operands.  The public operation instead maps its *leading* batch
    # axis, so flatten all remaining axes explicitly.
    output = kernel_mat @ value.reshape(value.shape[0], -1)
    return output.reshape(*kernel_mat.shape[:-1], *value.shape[1:])


def apply_kernel_to_joint_state(joint_state, kernel_mat: torch.Tensor):
    """Apply a left kernel to every materialized state/batch channel."""
    if not isinstance(kernel_mat, torch.Tensor) or kernel_mat.ndim < 2:
        raise ValueError("kernel_mat must be a rank-two-or-greater tensor")
    return _new_state(
        joint_state,
        **{name: _apply_kernel(kernel_mat, getattr(joint_state, name)) for name in _CHANNELS},
        dt=_apply_kernel(kernel_mat, joint_state.dt) if joint_state.dt is not None else None,
    )


def scale_joint_state(joint_state, dt):
    """Scale derivatives for a dimensionless time factor without mutating input."""
    scale = torch.as_tensor(dt, device=joint_state.device, dtype=joint_state.dtype)
    def scale_for(value):
        result = scale
        while result.ndim < value.ndim:
            result = result.unsqueeze(-1)
        return result

    return _new_state(
        joint_state,
        velocity=None if joint_state.velocity is None else joint_state.velocity * scale_for(joint_state.velocity),
        acceleration=None if joint_state.acceleration is None else joint_state.acceleration * scale_for(joint_state.acceleration).square(),
        jerk=None if joint_state.jerk is None else joint_state.jerk * scale_for(joint_state.jerk).pow(3),
    )


def scale_joint_state_by_dt(joint_state, dt, new_dt):
    """Rescale derivatives from ``dt`` to ``new_dt`` while retaining metadata."""
    old = torch.as_tensor(dt, device=joint_state.device, dtype=joint_state.dtype)
    new = torch.as_tensor(new_dt, device=joint_state.device, dtype=joint_state.dtype)
    if bool((old == 0).any().item()) or bool((new == 0).any().item()):
        raise ValueError("dt and new_dt must be non-zero")
    ratio = old / new
    def scale_for(value):
        output = ratio
        while output.ndim < value.ndim:
            output = output.unsqueeze(-1)
        return output
    return _new_state(
        joint_state,
        velocity=None if joint_state.velocity is None else joint_state.velocity * scale_for(joint_state.velocity),
        acceleration=None if joint_state.acceleration is None else joint_state.acceleration * scale_for(joint_state.acceleration).square(),
        jerk=None if joint_state.jerk is None else joint_state.jerk * scale_for(joint_state.jerk).pow(3),
        dt=new,
        knot_dt=None if joint_state.knot_dt is None else joint_state.knot_dt * (new / old),
    )


def scale_joint_state_time(joint_state, new_dt):
    if joint_state.dt is None:
        raise ValueError("joint_state.dt is required")
    return scale_joint_state_by_dt(joint_state, joint_state.dt, new_dt)


def calculate_fd_from_position(joint_state, dt=None):
    """Populate finite-difference derivatives in place, preserving autograd."""
    step = joint_state.dt if dt is None else dt
    if step is None:
        raise ValueError("dt is required")
    joint_state.velocity = fd_tensor(joint_state.position, step)
    joint_state.acceleration = fd_tensor(joint_state.velocity, step)
    joint_state.jerk = fd_tensor(joint_state.acceleration, step)
    return joint_state


def reorder_joint_state(joint_state, ordered_joint_names):
    return joint_state.reorder(list(ordered_joint_names))


def reindex_joint_state_inplace(joint_state, joint_names):
    """Reorder state channels in place and return the same object."""
    value = joint_state.reorder(list(joint_names))
    return joint_state.copy_reference(value)


def augment_joint_state(joint_state, joint_names, lock_joints=None):
    """Fill a requested joint ordering from active plus optional locked joints."""
    names = list(joint_names)
    if joint_state.joint_names is None:
        raise ValueError("joint_names required")
    if len(set(names)) != len(names):
        raise ValueError("joint_names must not contain duplicates")
    if lock_joints is None:
        return reorder_joint_state(joint_state, names)
    if lock_joints.joint_names is None:
        raise ValueError("lock_joints.joint_names required")
    overlap = set(joint_state.joint_names).intersection(lock_joints.joint_names)
    if overlap:
        raise ValueError("lock_joints is also listed in js.joint_names")
    available = set(joint_state.joint_names).union(lock_joints.joint_names)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"requested joints are absent: {missing}")
    return reorder_joint_state(append_joints_to_state(joint_state, lock_joints), names)


def cat_many(states):
    if not states:
        raise ValueError("states must not be empty")
    output = states[0]
    for state in states[1:]:
        output = cat_joint_states(output, state, -1)
    return output


def append_joints_to_state(joint_state, other_js):
    """Append joint channels, broadcasting a locked-state over trajectory axes.

    cuRobo commonly keeps locked joints as a single ``[D_locked]`` state and
    appends it to active state tensors shaped ``[B, D]`` or ``[B, H, D]``.
    The CUDA implementation handles that through a mix of JIT repeat and
    buffer writes.  This direct PyTorch version gives the same public layout
    while preserving explicit derivative channels and autograd on CPU/MPS.

    Raw packed-knot append remains an explicit boundary: upstream itself
    raises for it and there is no safe universal relation between independent
    B-spline control-point layouts.
    """
    from .state_joint import JointState

    if not joint_state.joint_names or not other_js.joint_names:
        raise ValueError("joint_names are required to append")
    if joint_state.position.device != other_js.position.device:
        raise ValueError("appended JointStates must be on the same device")
    if joint_state.position.dtype != other_js.position.dtype:
        raise ValueError("appended JointStates must share dtype")
    if joint_state.knot is not None and other_js.knot is not None:
        raise NotImplementedError("knot append requires a shared knot layout")

    target_shape = tuple(joint_state.position.shape)
    prefix = target_shape[:-1]

    def broadcast(value, *, channel: str):
        if value is None:
            return None
        if value.ndim == 0 or value.shape[-1] != other_js.position.shape[-1]:
            raise ValueError(f"{channel} must end in the appended DOF dimension")
        try:
            return value.expand(*prefix, value.shape[-1])
        except RuntimeError as error:
            raise ValueError(
                "appending joints requires the new joints to have a shape matching "
                "the current batch/trajectory prefix or singleton dimensions"
            ) from error

    right_position = broadcast(other_js.position, channel="position")

    def append_channel(channel: str):
        left = getattr(joint_state, channel)
        right = broadcast(getattr(other_js, channel), channel=channel)
        if left is None and right is None:
            return None
        if left is None:
            left = torch.zeros_like(joint_state.position)
        if left.shape != joint_state.position.shape:
            raise ValueError(f"{channel} must match the active position layout")
        if right is None:
            right = torch.zeros_like(right_position)
        return torch.cat((left, right), dim=-1)

    result = JointState(
        position=torch.cat((joint_state.position, right_position), dim=-1),
        velocity=append_channel("velocity"),
        acceleration=append_channel("acceleration"),
        jerk=append_channel("jerk"),
        joint_names=joint_state.joint_names + other_js.joint_names,
        device_cfg=joint_state.device_cfg,
        # Pinned cuRobo gives a supplied locked-state timing value precedence.
        dt=_clone(other_js.dt) if other_js.dt is not None else _clone(joint_state.dt),
        aux_data=dict(joint_state.aux_data),
        knot=_clone(joint_state.knot),
        knot_dt=_clone(joint_state.knot_dt),
        control_space=joint_state.control_space,
    )
    return result
