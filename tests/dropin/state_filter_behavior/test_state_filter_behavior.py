"""Behavioral coverage for the portable pinned ``state_filter`` facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.state_filter import FilterCfg, JointStateFilter


def _cfg(
    control_space: ControlSpace = ControlSpace.ACCELERATION,
    *,
    device: str = "cpu",
    dt: float = 0.1,
    teleport: bool = False,
) -> FilterCfg:
    return FilterCfg.create(
        {"position": 0.25, "velocity": 0.5, "acceleration": 0.75, "jerk": 1.0},
        dt=dt,
        control_space=control_space,
        device_cfg=DeviceCfg(device),
        teleport_mode=teleport,
    )


def _state(device: str = "cpu") -> JointState:
    position = torch.zeros(2, 3, device=device)
    return JointState(
        position,
        torch.ones_like(position),
        torch.full_like(position, 2.0),
        ["j0", "j1", "j2"],
        torch.full_like(position, 3.0),
        DeviceCfg(device),
    )


def test_filter_blends_all_channels_is_persistent_and_autograd_safe() -> None:
    filter_ = JointStateFilter(_cfg())
    first = _state()
    held_buffer = filter_.filter_joint_state(first)
    raw_position = torch.full((2, 3), 4.0, requires_grad=True)
    second = JointState(
        raw_position,
        torch.full_like(raw_position, 5.0),
        torch.full_like(raw_position, 6.0),
        ["j0", "j1", "j2"],
        torch.full_like(raw_position, 7.0),
    )
    result = filter_.filter_joint_state(second)

    assert result is held_buffer
    torch.testing.assert_close(result.position, torch.ones(2, 3))
    torch.testing.assert_close(result.velocity, torch.full((2, 3), 3.0))
    torch.testing.assert_close(result.acceleration, torch.full((2, 3), 5.0))
    torch.testing.assert_close(result.jerk, torch.full((2, 3), 7.0))
    result.position.square().sum().backward()
    assert raw_position.grad is not None
    torch.testing.assert_close(raw_position.grad, torch.full((2, 3), 0.5))


def test_filter_disabled_is_identity_and_reset_allows_shape_reconfiguration() -> None:
    disabled = JointStateFilter(FilterCfg.create({}, enable=False))
    state = _state()
    assert disabled.filter_joint_state(state) is state

    filter_ = JointStateFilter(_cfg())
    filter_.filter_joint_state(state)
    with pytest.raises(ValueError, match="call reset"):
        filter_.filter_joint_state(JointState.from_position(torch.zeros(1, 3)))
    filter_.reset()
    changed = filter_.filter_joint_state(JointState.from_position(torch.zeros(1, 3)))
    assert changed.position.shape == (1, 3)


def test_integrators_are_batched_broadcastable_and_do_not_alias_explicit_state() -> None:
    acceleration = JointStateFilter(_cfg(ControlSpace.ACCELERATION, dt=0.5))
    external = _state()
    output = acceleration.integrate_acc(torch.tensor([2.0, 4.0, 6.0]), external)
    assert output is not acceleration.cmd_joint_state
    torch.testing.assert_close(external.position, torch.zeros(2, 3))
    torch.testing.assert_close(output.acceleration, torch.tensor([[2.0, 4.0, 6.0]]).expand(2, -1))
    torch.testing.assert_close(output.velocity, torch.tensor([[2.0, 3.0, 4.0]]).expand(2, -1))
    torch.testing.assert_close(output.position, torch.tensor([[1.0, 1.5, 2.0]]).expand(2, -1))
    torch.testing.assert_close(output.jerk, torch.zeros(2, 3))

    jerk = JointStateFilter(_cfg(ControlSpace.ACCELERATION, dt=0.5))
    jerk_result = jerk.integrate_jerk(2.0, _state())
    torch.testing.assert_close(jerk_result.acceleration, torch.full((2, 3), 3.0))
    torch.testing.assert_close(jerk_result.velocity, torch.full((2, 3), 2.5))
    torch.testing.assert_close(jerk_result.position, torch.full((2, 3), 1.25))
    torch.testing.assert_close(jerk_result.jerk, torch.full((2, 3), 2.0))

    velocity = JointStateFilter(_cfg(ControlSpace.VELOCITY, dt=0.5))
    velocity_result = velocity.integrate_action(torch.tensor([[2.0], [4.0]]), _state())
    torch.testing.assert_close(velocity_result.position, torch.tensor([[1.0] * 3, [2.0] * 3]))


def test_position_teleport_dt_and_action_validation_boundaries() -> None:
    filter_ = JointStateFilter(_cfg(ControlSpace.POSITION, dt=0.5))
    result = filter_.integrate_pos(torch.full((2, 3), 2.0), _state())
    torch.testing.assert_close(result.velocity, torch.full((2, 3), 4.0))

    teleport = JointStateFilter(_cfg(ControlSpace.POSITION, dt=0.0, teleport=True))
    teleported = teleport.integrate_action(torch.full((2, 3), 9.0), _state())
    torch.testing.assert_close(teleported.position, torch.full((2, 3), 9.0))
    torch.testing.assert_close(teleported.velocity, torch.ones(2, 3))

    invalid_dt = JointStateFilter(_cfg(ControlSpace.POSITION, dt=0.0))
    with pytest.raises(ValueError, match="nonzero"):
        invalid_dt.integrate_pos(torch.ones(2, 3), _state())
    with pytest.raises(ValueError, match="final dimension"):
        filter_.integrate_pos(torch.ones(2, 2), _state())
    with pytest.raises(ValueError, match="required"):
        JointStateFilter(_cfg()).integrate_acc(torch.ones(3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_filter_and_integrators_stay_on_mps_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    filter_ = JointStateFilter(_cfg(device="mps", dt=0.2))
    result = filter_.filter_joint_state(_state("mps"))
    assert result.position.device.type == "mps"
    action_result = filter_.integrate_acc(torch.ones(3, device="mps"))
    assert action_result.position.device.type == "mps"
    torch.testing.assert_close(action_result.position.cpu(), torch.full((2, 3), 0.24))
