"""Portable lifecycle checks for pinned motion-planning result values."""

from __future__ import annotations

import pytest
import torch

from curobo._src.motion.motion_planner_result import GraspPlanResult, MotionPlannerResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _result(device: str = "cpu") -> GraspPlanResult:
    tensor_device = torch.device(device)
    trajectory = JointState.from_position(
        torch.arange(24, device=tensor_device, dtype=torch.float32).reshape(3, 2, 4),
        ["joint_0", "joint_1", "joint_2", "joint_3"],
    )
    output = GraspPlanResult(
        success=torch.tensor([[True, False], [False, False], [True, True]], device=tensor_device),
        approach_success=torch.tensor(
            [[True, False], [False, False], [True, True]], device=tensor_device
        ),
        grasp_success=torch.tensor(
            [[True, False], [False, False], [True, True]], device=tensor_device
        ),
        lift_success=torch.tensor(
            [[True, False], [False, False], [True, True]], device=tensor_device
        ),
        approach_trajectory=trajectory,
        approach_trajectory_dt=torch.tensor([0.1, 0.2, 0.3], device=tensor_device),
        approach_interpolated_last_tstep=torch.tensor([1, 2, 3], device=tensor_device),
        goalset_index=torch.tensor([[2], [0], [1]], device=tensor_device, dtype=torch.long),
        status="partial",
        planning_time=0.125,
    )
    output.stage_payload = MotionPlannerResult(output.success.clone())
    output.debug_info = {"cost": torch.tensor([3.0, 5.0, 7.0], device=tensor_device)}
    return output


def test_validate_selection_and_success_filter_are_batch_stable():
    result = _result().validate()
    assert result.num_success == 2
    assert result.num_failures == 1
    assert result.failure_mask.tolist() == [False, True, False]

    selected = result.select_batch(2)
    assert selected.batch_size == 1
    assert selected.success.shape == (1, 2)
    assert selected.approach_trajectory.position.shape == (1, 2, 4)
    assert selected.approach_trajectory_dt.shape == (1,)
    assert selected.goalset_index.tolist() == [[1]]
    assert selected.stage_payload.success.shape == (1, 2)
    assert selected.debug_info["cost"].tolist() == [7.0]

    successful = result.successful()
    assert successful.success.shape == (2, 2)
    assert successful.goalset_index.tolist() == [[2], [1]]
    assert successful.approach_trajectory.position.shape == (2, 2, 4)
    assert successful.debug_info["cost"].tolist() == [3.0, 7.0]

    reversed_rows = result.select_batch([-1, 0])
    assert reversed_rows.goalset_index.tolist() == [[1], [2]]
    selected.success[0, 0] = False
    assert bool(result.success[2, 0])


def test_validate_reports_shape_device_dtype_and_time_errors():
    result = _result()
    result.grasp_success = torch.ones((3, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="grasp_success"):
        result.validate()

    result = _result()
    result.goalset_index = torch.zeros((2, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="goalset_index"):
        result.validate()

    result = _result()
    result.planning_time = -0.1
    with pytest.raises(ValueError, match="planning_time"):
        result.validate()

    with pytest.raises(ValueError, match="batched success"):
        MotionPlannerResult(success=torch.tensor(True)).successful()


def test_device_move_detach_and_boolean_batch_selection_preserve_result_dtypes():
    result = _result()
    result.payload = torch.ones(3, requires_grad=True) * 2
    detached = result.detach()
    assert not detached.payload.requires_grad
    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.approach_trajectory.position.dtype is torch.float64
    assert moved.success.dtype is torch.bool
    assert moved.goalset_index.dtype is torch.long

    selected = result.select_batch(torch.tensor([True, False, True]))
    assert selected.success.tolist() == [[True, False], [True, True]]
    with pytest.raises(IndexError, match="boolean batch index"):
        result.select_batch(torch.tensor([True, False]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_result_selection_runs_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = _result("mps").validate()
    selected = result.successful().to(DeviceCfg(torch.device("mps"), torch.float32))
    assert selected.device.type == "mps"
    assert selected.approach_trajectory.device.type == "mps"
    assert selected.goalset_index.dtype is torch.long
