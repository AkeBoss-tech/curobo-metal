"""High-value portable lifecycle checks for the pinned ``JointState`` surface."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


def _state(*, device: str = "cpu") -> JointState:
    position = torch.arange(12.0, device=device).reshape(2, 2, 3).requires_grad_()
    return JointState(
        position,
        torch.full_like(position, 1.0),
        torch.full_like(position, 2.0),
        ["a", "b", "c"],
        torch.full_like(position, 3.0),
        DeviceCfg(device),
        dt=torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device),
        knot=position * 0.25,
        knot_dt=torch.tensor([0.5, 0.6], device=device),
        aux_data={"producer": "test"},
        control_space=ControlSpace.POSITION,
    )


def test_joint_state_derivative_channels_are_independent_and_packing_is_fixed_width() -> None:
    state = JointState.from_position(torch.ones(2, 3), ["a", "b", "c"])
    assert state.velocity.data_ptr() != state.acceleration.data_ptr()
    assert state.velocity.data_ptr() != state.jerk.data_ptr()
    state.velocity[0, 0] = 9.0
    assert state.acceleration[0, 0] == 0.0
    assert state.jerk[0, 0] == 0.0

    partial = JointState(torch.ones(2, 3), joint_names=["a", "b", "c"])
    packed = partial.get_state_tensor()
    assert packed.shape == (2, 12)
    torch.testing.assert_close(packed[..., :3], partial.position)
    torch.testing.assert_close(packed[..., 3:], torch.zeros(2, 9))


def test_joint_state_clone_to_copy_and_index_preserve_materialized_metadata() -> None:
    state = _state()
    clone = state.clone()
    assert clone.position.data_ptr() != state.position.data_ptr()
    assert clone.knot.data_ptr() != state.knot.data_ptr()
    assert clone.knot_dt.data_ptr() != state.knot_dt.data_ptr()
    assert clone.dt.data_ptr() != state.dt.data_ptr()
    assert clone.aux_data == state.aux_data and clone.aux_data is not state.aux_data
    assert clone.control_space is ControlSpace.POSITION

    target = _state()
    source = _state()
    source.knot.data.add_(10)
    source.knot_dt.add_(10)
    source.dt.add_(10)
    source.aux_data["producer"] = "source"
    source.control_space = ControlSpace.VELOCITY
    with torch.no_grad():
        target.copy_(source)
    torch.testing.assert_close(target.knot, source.knot)
    torch.testing.assert_close(target.knot_dt, source.knot_dt)
    torch.testing.assert_close(target.dt, source.dt)
    assert target.aux_data == {"producer": "source"}
    assert target.control_space is ControlSpace.VELOCITY

    with torch.no_grad():
        target[0] = source[1]
    torch.testing.assert_close(target.knot[0], source.knot[1])
    torch.testing.assert_close(target.knot_dt[0], source.knot_dt[1])
    torch.testing.assert_close(target.dt[0], source.dt[1])

    converted = state.to(DeviceCfg("cpu", torch.float64))
    assert converted.dtype is torch.float64
    assert converted.knot.dtype is torch.float64
    assert converted.knot_dt.dtype is torch.float64
    assert converted.dt.dtype is torch.float64


def test_joint_state_cat_and_stack_keep_names_timing_and_autograd() -> None:
    left = _state()
    right = _state()
    appended = left.cat(right.index_dof(torch.tensor([0])), -1)
    assert appended.position.shape == (2, 2, 4)
    assert appended.joint_names == ["a", "b", "c", "a"]
    torch.testing.assert_close(appended.dt, left.dt)
    torch.testing.assert_close(appended.knot, left.knot)
    assert appended.control_space is ControlSpace.POSITION

    stacked = left.stack(right)
    # Pinned V2's historical ``stack`` name concatenates trajectory waypoints
    # on the second-to-last axis; it does not create an extra seed dimension.
    assert stacked.position.shape == (2, 4, 3)
    assert stacked.joint_names == ["a", "b", "c"]
    (stacked.position.square().sum() + appended.position.sum()).backward()
    assert left.position.grad is not None and torch.isfinite(left.position.grad).all()
    assert right.position.grad is not None and torch.isfinite(right.position.grad).all()


def test_joint_state_copy_shape_boundary_and_mps_lifecycle_when_available(monkeypatch) -> None:
    target = JointState.from_position(torch.zeros(2, 3))
    source = JointState.from_position(torch.zeros(3, 3))
    with pytest.raises(ValueError, match="current state"):
        target.copy_(source, allow_clone=False)

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    state = _state(device="mps")
    clone = state.clone().detach().to(DeviceCfg("mps"))
    assert clone.position.device.type == "mps"
    assert clone.dt.device.type == "mps"
    assert clone.knot.device.type == "mps"
