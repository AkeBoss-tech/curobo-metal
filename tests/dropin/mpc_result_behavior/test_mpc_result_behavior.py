"""Portable lifecycle coverage for ``MPCSolverResult``."""

from __future__ import annotations

import pytest
import torch

from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


def _state(values: torch.Tensor) -> JointState:
    return JointState(
        values,
        velocity=values.add(100),
        acceleration=values.add(200),
        jerk=values.add(300),
        dt=torch.full(values.shape[:2], 0.1, device=values.device),
        joint_names=["a", "b", "c"],
    )


def _result(device: str = "cpu") -> MPCSolverResult:
    actions = torch.arange(24, device=device, dtype=torch.float32).reshape(2, 4, 3)
    action_state = _state(actions)
    command = JointState.from_position(actions[:, 0].clone(), ["a", "b", "c"])
    robot = RobotState(_state(actions.clone()), joint_torque=actions.add(400))
    return MPCSolverResult(
        success=torch.tensor([True, False], device=device),
        solution=actions.add(10),
        js_solution=action_state.clone(),
        cspace_error=torch.tensor([0.1, 0.2], device=device),
        seed_cost=torch.tensor([1.0, 2.0], device=device),
        goalset_index=torch.tensor([4, 9], device=device),
        metrics={"per_batch": torch.tensor([3.0, 5.0], device=device), "nested": {"value": actions.clone()}},
        batch_size=2,
        next_action=command,
        action_sequence=action_state,
        full_action_sequence=action_state.clone(),
        robot_state_sequence=robot,
        action_buffer=actions.clone(),
        action_dt=0.1,
    )


def test_clone_deep_copies_actions_robot_and_nested_diagnostics() -> None:
    result = _result()
    clone = result.clone()
    clone.action_buffer[0, 0, 0] = -1
    clone.action_sequence.velocity[0, 0, 0] = -2
    clone.robot_state_sequence.joint_torque[0, 0, 0] = -3
    clone.metrics["nested"]["value"][0, 0, 0] = -4

    assert result.action_buffer[0, 0, 0].item() == 0
    assert result.action_sequence.velocity[0, 0, 0].item() == 100
    assert result.robot_state_sequence.joint_torque[0, 0, 0].item() == 400
    assert result.metrics["nested"]["value"][0, 0, 0].item() == 0


def test_action_batch_selection_and_successful_preserve_batch_ranks() -> None:
    result = _result()
    selected = result.select_batch(1)
    assert selected.batch_size == 1
    assert selected.success.tolist() == [False]
    assert selected.action_buffer.shape == (1, 4, 3)
    assert selected.action_sequence.position.shape == (1, 4, 3)
    assert selected.robot_state_sequence.joint_state.position.shape == (1, 4, 3)
    assert selected.metrics["per_batch"].tolist() == [5.0]

    successful = result.successful()
    assert successful.batch_size == 1
    assert successful.success.tolist() == [True]
    assert successful.action_buffer.shape == (1, 4, 3)

    with pytest.raises(IndexError, match="outside"):
        result.select_batch(2)


def test_action_at_prefers_current_buffer_and_preserves_sequence_channels() -> None:
    result = _result()
    buffered = result.action_at(2)
    torch.testing.assert_close(buffered.position, result.action_buffer[:, 2])

    result.action_buffer = None
    action = result.action_at(3)
    torch.testing.assert_close(action.position, result.action_sequence.position[:, 3])
    torch.testing.assert_close(action.velocity, result.action_sequence.velocity[:, 3])
    assert action.dt.shape == (2,)
    assert result.action_horizon == 4

    with pytest.raises(IndexError, match="outside"):
        result.action_at(4)


def test_to_preserves_index_dtype_and_moves_portable_state_payloads() -> None:
    result = _result()
    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.device == torch.device("cpu")
    assert moved.dtype == torch.float64
    assert moved.action_buffer.dtype == torch.float64
    assert moved.success.dtype == torch.bool
    assert moved.goalset_index.dtype == torch.int64
    assert moved.action_sequence.velocity.dtype == torch.float64
    assert moved.robot_state_sequence.joint_torque.dtype == torch.float64


def test_action_layout_validation_rejects_bad_contracts() -> None:
    result = _result()
    result.validate_action_layout()
    result.action_buffer = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="action_buffer"):
        result.validate_action_layout()

    result = _result()
    result.action_dt = 0.0
    with pytest.raises(ValueError, match="action_dt"):
        result.validate_action_layout()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_mpc_result_lifecycle_has_no_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = _result("mps")
    result.validate_action_layout()
    selected = result.select_batch([0])
    action = selected.action_at(1)
    moved = result.to(DeviceCfg(torch.device("mps"), torch.float32))

    assert selected.action_buffer.device.type == "mps"
    assert selected.robot_state_sequence.joint_torque.device.type == "mps"
    assert action.velocity.device.type == "mps"
    assert moved.action_sequence.position.device.type == "mps"
