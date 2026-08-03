"""Pinned state-base and filter-coefficient compatibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import is_dataclass

import pytest
import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_base import State
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import blend_joint_states


def test_state_is_dataclass_abstract_sequence_and_joint_state_is_a_state() -> None:
    assert is_dataclass(State)
    assert issubclass(State, Sequence)
    with pytest.raises(TypeError):
        State()

    joint_state = JointState.from_position(torch.zeros(2, 3))
    assert isinstance(joint_state, State)
    assert isinstance(joint_state, Sequence)


def test_filter_coeff_defaults_are_pinned_mutable_scalar_values() -> None:
    coeff = FilterCoeff()
    assert (coeff.position, coeff.velocity, coeff.acceleration, coeff.jerk) == (0.0, 0.0, 0.0, 0.0)

    coeff.position = -0.25
    coeff.velocity = 1.0
    assert coeff.position == -0.25
    assert coeff.velocity == 1.0


def test_filter_coeff_blends_each_channel_without_host_coefficient_tensors() -> None:
    target = JointState.from_position(torch.zeros(2, 3))
    incoming_position = torch.full((2, 3), 4.0, requires_grad=True)
    incoming = JointState.from_position(incoming_position)

    blend_joint_states(target, incoming, FilterCoeff(position=0.25))
    torch.testing.assert_close(target.position, torch.ones(2, 3))
    target.position.square().sum().backward()
    torch.testing.assert_close(incoming_position.grad, torch.full((2, 3), 0.5))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_filter_coeff_blending_stays_on_mps_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    target = JointState.from_position(torch.zeros(2, 3, device="mps"))
    incoming = JointState.from_position(torch.full((2, 3), 4.0, device="mps", requires_grad=True))

    blend_joint_states(target, incoming, FilterCoeff(position=0.25))
    assert target.position.device.type == "mps"
    target.position.square().sum().backward()
    assert incoming.position.grad is not None
