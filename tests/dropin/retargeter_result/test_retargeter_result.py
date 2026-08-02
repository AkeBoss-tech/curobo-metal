"""Lifecycle coverage for the pinned V2 retargeting result surface."""

import pytest
import torch

from curobo._src.motion.motion_retargeter_result import RetargetResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _state(position: torch.Tensor, names=None) -> JointState:
    return JointState(
        position=position,
        velocity=position + 10,
        acceleration=position + 20,
        jerk=position + 30,
        dt=torch.full(position.shape[:2], 0.1, device=position.device, dtype=position.dtype),
        joint_names=names or ["a", "b", "c", "d"],
    )


def _result(device: str = "cpu", *, sequence: bool = True) -> RetargetResult:
    shape = (2, 3, 4) if sequence else (2, 4)
    position = torch.arange(int(torch.tensor(shape).prod()), device=device, dtype=torch.float32).reshape(shape)
    state = _state(position.requires_grad_(True))
    trajectory = _state(torch.arange(40, device=device, dtype=torch.float32).reshape(2, 5, 4))
    return RetargetResult(state, trajectory)


def test_pinned_fields_and_portable_result_metadata():
    result = _result()
    assert result.batch_size == 2
    assert result.num_dof == 4
    assert result.num_frames == 3
    assert result.num_trajectory_frames == 5
    assert result.is_sequence and result.is_mpc
    assert result.device.type == "cpu"
    assert result.dtype is torch.float32

    frame = _result(sequence=False)
    assert frame.num_frames == 1
    assert not frame.is_sequence


def test_clone_detach_and_to_are_isolated_and_device_safe():
    result = _result()
    clone = result.clone()
    clone.joint_state.position[0, 0, 0] = -1
    clone.trajectory.velocity[0, 0, 0] = -1
    assert result.joint_state.position[0, 0, 0].item() == 0.0
    assert result.trajectory.velocity[0, 0, 0].item() == 10.0

    detached = result.detach()
    assert not detached.joint_state.position.requires_grad
    assert not detached.trajectory.position.requires_grad
    assert result.joint_state.position.requires_grad

    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.joint_state.position.dtype is torch.float64
    assert moved.trajectory.position.dtype is torch.float64
    assert moved.joint_state.joint_names == result.joint_state.joint_names


def test_batch_selection_preserves_batch_rank_and_derivative_streams():
    result = _result()
    selected = result[1]
    assert selected.joint_state.position.shape == (1, 3, 4)
    assert selected.trajectory.position.shape == (1, 5, 4)
    assert selected.joint_state.velocity[0, 0, 0].item() == 22.0
    assert selected.joint_state.dt.shape == (1, 3)

    selected.joint_state.position[0, 0, 0] = -2
    assert result.joint_state.position[1, 0, 0].item() == 12.0
    mask = result.select_batch(torch.tensor([True, False]))
    assert mask.joint_state.position.shape == (1, 3, 4)
    assert result[-1].joint_state.position[0, 0, 0].item() == 12.0


def test_validation_rejects_incompatible_materialized_states():
    with pytest.raises(TypeError, match="joint_state"):
        RetargetResult(torch.zeros(1, 4))

    state = JointState.from_position(torch.zeros(1, 4), ["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="intermediate-frame"):
        RetargetResult(state, JointState.from_position(torch.zeros(1, 4)))
    with pytest.raises(ValueError, match="environment batch"):
        RetargetResult(state, JointState.from_position(torch.zeros(2, 1, 4)))
    with pytest.raises(IndexError, match="out of range"):
        RetargetResult(state)[1]
    with pytest.raises(IndexError, match="boolean"):
        _result().select_batch(torch.tensor([True]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_lifecycle_remains_materialized_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = _result("mps")
    selected = result.select_batch([1])
    detached = selected.detach()
    assert selected.device.type == "mps"
    assert selected.trajectory.device.type == "mps"
    assert not detached.joint_state.position.requires_grad
    assert selected.to("mps").trajectory.device.type == "mps"
