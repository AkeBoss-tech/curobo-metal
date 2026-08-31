"""Shared JointState replay probe used by portable and pinned CUDA runners."""

from __future__ import annotations

from typing import Any

import numpy as np


def run(raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Exercise observable JointState values, operations, metadata, and VJPs."""
    import torch
    from curobo._src.state.state_joint import JointState
    from curobo._src.state.state_joint_ops import (
        apply_kernel_to_joint_state,
        cat_joint_states,
        joint_state_to_tensor,
        reorder_joint_state,
        repeat_joint_state_seeds,
        scale_joint_state,
    )

    def tensor(name: str, *, grad: bool = False) -> Any:
        value = torch.as_tensor(raw[name], device=device, dtype=torch.float32)
        return value.clone().detach().requires_grad_(grad)

    position = tensor("joint_position", grad=True)
    state = JointState(
        position=position,
        velocity=tensor("joint_velocity"),
        acceleration=tensor("joint_acceleration"),
        jerk=tensor("joint_jerk"),
        joint_names=["j0", "j1"],
        dt=tensor("joint_dt"),
    )
    packed = joint_state_to_tensor(state)
    reordered = reorder_joint_state(state, ["j1", "j0"])
    repeated = repeat_joint_state_seeds(state, 3)
    kernel_state = apply_kernel_to_joint_state(state, tensor("joint_kernel"))
    scaled = scale_joint_state(state, torch.tensor(0.5, device=device))
    concatenated = cat_joint_states(state, state, -1)

    packed_weights = torch.arange(
        packed.numel(), device=device, dtype=packed.dtype
    ).reshape_as(packed)
    reorder_weights = torch.tensor(
        [[0.25, -0.5], [0.75, 1.25]], device=device, dtype=packed.dtype
    )
    loss = (packed * packed_weights).sum() + (reordered.position * reorder_weights).sum()
    (input_gradient,) = torch.autograd.grad(loss, position)

    source = tensor("joint_noncontiguous_source")
    noncontiguous = source[:, ::2]
    if noncontiguous.is_contiguous():
        raise RuntimeError("joint-state edge corpus unexpectedly became contiguous")
    noncontiguous_state = JointState.from_position(noncontiguous, ["j0", "j1"])

    alias_state = JointState.from_position(tensor("joint_position"), ["j0", "j1"])
    acceleration_before = alias_state.acceleration.clone()
    alias_state.velocity.add_(1.0)
    derivative_channels_independent = int(
        bool(torch.equal(alias_state.acceleration, acceleration_before))
    )
    clone = alias_state.clone()
    original_before = alias_state.position.clone()
    clone.position.add_(1.0)
    clone_independent = int(bool(torch.equal(alias_state.position, original_before)))

    def array(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    return {
        "packed": array(packed),
        "reordered_position": array(reordered.position),
        "reordered_velocity": array(reordered.velocity),
        "repeated_position": array(repeated.position),
        "repeated_dt": array(repeated.dt),
        "kernel_position": array(kernel_state.position),
        "kernel_velocity": array(kernel_state.velocity),
        "kernel_dt": array(kernel_state.dt),
        "scaled_velocity": array(scaled.velocity),
        "scaled_acceleration": array(scaled.acceleration),
        "scaled_jerk": array(scaled.jerk),
        "concatenated_position": array(concatenated.position),
        "input_gradient": array(input_gradient),
        "noncontiguous_packed": array(joint_state_to_tensor(noncontiguous_state)),
        "derivative_channels_independent": np.asarray(
            [derivative_channels_independent], dtype=np.int8
        ),
        "clone_independent": np.asarray([clone_independent], dtype=np.int8),
    }
