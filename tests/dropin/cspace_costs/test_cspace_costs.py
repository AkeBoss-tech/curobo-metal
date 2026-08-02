"""Portable tensor coverage for the pinned c-space cost bridge signatures."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.wp_cspace_position import (
    PositionCSpaceFunction,
    forward_cspace_position_warp,
)
from curobo._src.cost.wp_cspace_state import StateCSpaceFunction, forward_cspace_state_warp


def _position_inputs(device: str = "cpu", *, use_grad_input: bool = True):
    batch, horizon, dof = 2, 2, 2
    position = torch.tensor(
        [[[1.2, 0.0], [0.5, -1.1]], [[0.1, 0.2], [0.4, 0.0]]],
        device=device,
        requires_grad=True,
    )
    torque = torch.tensor(
        [[[0.0, 2.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        device=device,
        requires_grad=True,
    )
    zeros = lambda: torch.zeros((batch, horizon, dof), device=device)
    return (
        position, torque,
        torch.tensor([[0.0, 0.0], [0.25, -0.25]], device=device),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device),
        torch.tensor([[-2.0, -2.0], [2.0, 2.0]], device=device),
        torch.tensor([1.0, 2.0], device=device), torch.tensor([0.0, 0.0], device=device),
        torch.tensor([0.5], device=device), torch.tensor([2.0, 1.0], device=device),
        torch.tensor([0.1, 0.2], device=device),
        torch.tensor([[0.0, 0.0], [0.25, -0.25]], device=device),
        torch.tensor([[0.1, -0.2], [0.0, 0.0]], device=device),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=device),
        torch.tensor([1.0, 0.5], device=device),
        zeros(), zeros(), zeros(), use_grad_input,
    )


def _state_inputs(device: str = "cpu", *, retime: bool = False):
    batch, horizon, dof = 2, 3, 1
    channels = [
        torch.tensor([[[1.2], [0.0], [0.5]], [[0.2], [0.1], [0.0]]], device=device, requires_grad=True),
        torch.tensor([[[0.3], [0.1], [0.0]], [[0.0], [0.0], [0.0]]], device=device, requires_grad=True),
        torch.zeros((batch, horizon, dof), device=device, requires_grad=True),
        torch.zeros((batch, horizon, dof), device=device, requires_grad=True),
        torch.tensor([[[0.0], [0.0], [0.0]], [[2.5], [0.0], [0.0]]], device=device, requires_grad=True),
    ]
    zeros = lambda: torch.zeros((batch, horizon, dof), device=device)
    return (
        *channels, torch.tensor([1.0, 0.5], device=device), torch.tensor([[0.0]], device=device),
        torch.tensor([0, 0], device=device, dtype=torch.int32),
        torch.tensor([[-1.0], [1.0]], device=device), torch.tensor([[-2.0], [2.0]], device=device),
        torch.tensor([[-3.0], [3.0]], device=device), torch.tensor([[-4.0], [4.0]], device=device),
        torch.tensor([[-2.0], [2.0]], device=device), torch.ones(5, device=device), torch.zeros(5, device=device),
        torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6], device=device), torch.tensor([0.5], device=device),
        torch.tensor([0.1], device=device), torch.tensor([2.0], device=device),
        zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), retime, retime, True,
    )


def test_position_function_matches_expected_terms_buffers_and_backward() -> None:
    values = _position_inputs()
    output = PositionCSpaceFunction.apply(*values)
    position, torque = values[:2]
    # First value includes target, dynamic velocity-tightened bound, and the
    # two V2 physical-time regularizers.
    torch.testing.assert_close(output[0, 0, 0], torch.tensor(1.878))
    # The direct Function preserves upstream's [B,H,D] output and diagnostics.
    assert output.shape == position.shape
    torch.testing.assert_close(values[16], output)
    assert values[17].abs().sum() > 0 and values[18].abs().sum() > 0
    (output * 0.25).sum().backward()
    torch.testing.assert_close(position.grad, values[17] * 0.25)
    torch.testing.assert_close(torque.grad, values[18] * 0.25)


def test_position_uses_positive_dt_velocity_tightening_and_rejects_bad_indices() -> None:
    values = list(_position_inputs())
    values[0] = torch.tensor([[[0.75, 0.0]], [[0.25, -0.25]]], requires_grad=True)
    values[1] = torch.zeros_like(values[0], requires_grad=True)
    values[16] = torch.zeros_like(values[0]); values[17] = torch.zeros_like(values[0]); values[18] = torch.zeros_like(values[0])
    output = PositionCSpaceFunction.apply(*values)
    # Base upper position bound is 1, but current 0 + velocity bound .5 at dt 1 tightens it to .5.
    torch.testing.assert_close(output[0, 0, 0], torch.tensor(0.664125))
    values[3] = torch.tensor([3, 0], dtype=torch.int32)
    with pytest.raises(ValueError, match="out-of-range"):
        PositionCSpaceFunction.apply(*values)


def test_state_function_target_terminal_factor_retime_and_all_channel_grads() -> None:
    values = _state_inputs(retime=True)
    output = StateCSpaceFunction.apply(*values)
    position, velocity, acceleration, jerk, torque = values[:5]
    assert output.shape == position.shape
    # Target is worth 10x less before terminal.  The first sample also has a
    # position bound and retimed velocity regularization contribution.
    torch.testing.assert_close(output[0, 0, 0], torch.tensor(0.1730))
    torch.testing.assert_close(output[0, -1, 0], torch.tensor(0.2500))
    output.sum().backward()
    for channel, diagnostic in zip((position, velocity, acceleration, jerk, torque), values[20:25]):
        torch.testing.assert_close(channel.grad, diagnostic)


@pytest.mark.parametrize("entrypoint", [forward_cspace_position_warp, forward_cspace_state_warp])
def test_raw_warp_kernel_entrypoints_remain_explicit_boundaries(entrypoint) -> None:
    with pytest.raises(NotImplementedError, match="CUDA/Warp"):
        entrypoint()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_position_and_state_functions_keep_buffers_and_grads_on_mps() -> None:
    position_values = _position_inputs("mps")
    position_cost = PositionCSpaceFunction.apply(*position_values)
    position_cost.sum().backward()
    assert position_cost.device.type == position_values[0].grad.device.type == "mps"
    state_values = _state_inputs("mps")
    state_cost = StateCSpaceFunction.apply(*state_values)
    state_cost.sum().backward()
    assert state_cost.device.type == state_values[0].grad.device.type == "mps"
