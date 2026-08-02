import pytest
import torch

from curobo._src.motion.motion_planner_result import (
    GraspPlanResult,
    MotionPlannerResult,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _result(device="cpu"):
    device = torch.device(device)
    trajectory = JointState.from_position(
        torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4),
        ["a", "b", "c", "d"],
    )
    result = GraspPlanResult(
        success=torch.tensor([[True, False], [False, False]], device=device),
        approach_success=torch.tensor([[True, False], [True, False]], device=device),
        grasp_success=torch.tensor([[True, False], [False, False]], device=device),
        lift_success=torch.tensor([[True, False], [False, False]], device=device),
        approach_trajectory=trajectory,
        approach_trajectory_dt=torch.tensor([0.1, 0.2], device=device),
        status="partial",
        planning_time=0.25,
        goalset_index=torch.tensor([[1], [0]], dtype=torch.long, device=device),
    )
    result.approach_result = MotionPlannerResult(result.approach_success)
    result.debug_info = {"per_problem": torch.tensor([3.0, 5.0], device=device)}
    return result


def test_result_lifecycle_clone_detach_and_metadata():
    result = _result()
    assert result.batch_size == 2
    assert result.device.type == "cpu"
    assert result.success_per_problem.tolist() == [True, False]
    assert result.num_success == 1
    assert result.success_ratio == pytest.approx(0.5)
    assert result.any_success() and not result.all_success()
    assert result.stage_success("approach").shape == (2, 2)
    assert result.stage_trajectory("approach") is result.approach_trajectory
    assert result.stage_trajectory("approach", interpolated=True) is None
    with pytest.raises(ValueError, match="stage"):
        result.stage_success("close")

    clone = result.clone()
    clone.success[0, 0] = False
    clone.approach_trajectory.position[0, 0, 0] = -1
    clone.debug_info["per_problem"][0] = -1
    assert bool(result.success[0, 0])
    assert result.approach_trajectory.position[0, 0, 0].item() == 0.0
    assert result.debug_info["per_problem"][0].item() == 3.0
    assert clone.approach_result is not result.approach_result

    differentiated = GraspPlanResult(success=torch.ones(1, dtype=torch.bool))
    differentiated.payload = (torch.ones(1, requires_grad=True) * 2.0)
    assert not differentiated.detach().payload.requires_grad


def test_result_selects_batch_and_nested_stage_metadata():
    result = _result()
    selected = result[1]
    assert selected.success.tolist() == [False, False]
    assert selected.approach_trajectory.position.shape == (3, 4)
    assert selected.approach_trajectory_dt.shape == ()
    assert selected.goalset_index.tolist() == [0]
    assert selected.approach_result.success.tolist() == [True, False]
    assert selected.debug_info["per_problem"].item() == 5.0
    with pytest.raises(IndexError, match="without a batched"):
        MotionPlannerResult()[0]


def test_result_to_preserves_bool_and_index_dtypes():
    result = _result().to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert result.success.dtype is torch.bool
    assert result.goalset_index.dtype is torch.long
    assert result.approach_trajectory.position.dtype is torch.float64
    assert result.approach_trajectory.device.type == "cpu"


def test_result_validation_fails_early_for_invalid_primary_fields():
    with pytest.raises(TypeError, match="success"):
        MotionPlannerResult(success=True)
    with pytest.raises(TypeError, match="dtype"):
        MotionPlannerResult(success=torch.ones(1))
    with pytest.raises(TypeError, match="status"):
        GraspPlanResult(status=3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_result_mps_move_and_batch_selection_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = _result("mps")
    assert result.device.type == "mps"
    assert result[0].approach_trajectory.device.type == "mps"
    moved = result.to(DeviceCfg(torch.device("mps"), torch.float32))
    assert moved.success.device.type == "mps"
    assert moved.goalset_index.dtype is torch.long
