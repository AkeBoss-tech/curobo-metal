"""Operations on portable JointState objects."""

from __future__ import annotations

import torch
from .filter_coeff import FilterCoeff
from curobo._src.util.tensor_util import fd_tensor


def blend_joint_states(target, new_state, coeff: FilterCoeff):
    for name in ("position", "velocity", "acceleration", "jerk"):
        old, new = getattr(target, name), getattr(new_state, name)
        if old is not None and new is not None:
            weight = getattr(coeff, name)
            old.copy_(weight * new + (1 - weight) * old)
    return target


def joint_state_to_tensor(joint_state):
    zero = torch.zeros_like(joint_state.position)
    return torch.cat([
        joint_state.position,
        joint_state.velocity if joint_state.velocity is not None else zero,
        joint_state.acceleration if joint_state.acceleration is not None else zero,
        joint_state.jerk if joint_state.jerk is not None else zero,
    ], dim=-1)


def stack_joint_states(js1, js2):
    from .state_joint import JointState
    return JointState.from_state_tensor(
        torch.cat((joint_state_to_tensor(js1), joint_state_to_tensor(js2)), dim=-2),
        js1.joint_names, js1.position.shape[-1],
    )


def cat_joint_states(js1, js2, dim: int):
    from .state_joint import JointState
    def cat(name):
        a, b = getattr(js1, name), getattr(js2, name)
        return None if a is None or b is None else torch.cat((a, b), dim=dim)
    names = js1.joint_names + js2.joint_names if dim == -1 else js1.joint_names
    return JointState(cat("position"), cat("velocity"), cat("acceleration"), names, cat("jerk"),
                      dt=js1.dt, control_space=js1.control_space)


def repeat_joint_state(joint_state, repeat_input):
    return joint_state._shape_map(lambda value: value.repeat(repeat_input))


def repeat_joint_state_seeds(joint_state, num_seeds: int):
    return joint_state.repeat_seeds(num_seeds)


def apply_kernel_to_joint_state(joint_state, kernel_mat):
    return joint_state._shape_map(lambda value: kernel_mat @ value)


def scale_joint_state(joint_state, dt):
    output = joint_state.clone()
    if output.velocity is not None: output.velocity = output.velocity * dt
    if output.acceleration is not None: output.acceleration = output.acceleration * dt**2
    if output.jerk is not None: output.jerk = output.jerk * dt**3
    return output


def scale_joint_state_by_dt(joint_state, dt, new_dt):
    output = joint_state.clone()
    ratio = torch.as_tensor(dt, device=output.device) / torch.as_tensor(new_dt, device=output.device)
    if output.velocity is not None: output.velocity *= ratio
    if output.acceleration is not None: output.acceleration *= ratio**2
    if output.jerk is not None: output.jerk *= ratio**3
    output.dt = new_dt
    return output


def scale_joint_state_time(joint_state, new_dt):
    if joint_state.dt is None:
        raise ValueError("joint_state.dt is required")
    return scale_joint_state_by_dt(joint_state, joint_state.dt, new_dt)


def calculate_fd_from_position(joint_state, dt=None):
    step = joint_state.dt if dt is None else dt
    if step is None:
        raise ValueError("dt is required")
    joint_state.velocity = fd_tensor(joint_state.position, step)
    joint_state.acceleration = fd_tensor(joint_state.velocity, step)
    joint_state.jerk = fd_tensor(joint_state.acceleration, step)
    return joint_state


def reorder_joint_state(joint_state, ordered_joint_names):
    return joint_state.reorder(ordered_joint_names)


def reindex_joint_state_inplace(joint_state, joint_names):
    value = joint_state.reorder(joint_names)
    return joint_state.copy_reference(value)


def augment_joint_state(joint_state, joint_names, lock_joints=None):
    if joint_state.joint_names is None:
        raise ValueError("joint_names required")
    sources = {name: joint_state[..., i:i+1] for i, name in enumerate(joint_state.joint_names)}
    if lock_joints is not None:
        sources.update({name: lock_joints[..., i:i+1] for i, name in enumerate(lock_joints.joint_names)})
    return cat_many([sources[name] for name in joint_names])


def cat_many(states):
    output = states[0]
    for state in states[1:]:
        output = cat_joint_states(output, state, -1)
    return output


def append_joints_to_state(joint_state, other_js):
    return cat_joint_states(joint_state, other_js, -1)
