"""Portable BaseSolverResult value/lifecycle coverage."""

import pytest
import torch

from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


def _result(device: str = "cpu", fill: float = 0.0) -> BaseSolverResult:
    batch, seeds, horizon, dof = 2, 3, 2, 2
    payload = torch.arange(batch * seeds * horizon * dof, device=device, dtype=torch.float32)
    payload = payload.reshape(batch, seeds, horizon, dof).add(fill)
    state = JointState(
        payload.clone(), velocity=payload.add(10), acceleration=payload.add(20),
        jerk=payload.add(30), dt=torch.full((batch, seeds), 0.1, device=device),
        joint_names=["a", "b"],
    )
    robot = RobotState(JointState(payload[..., 0, :].clone(), joint_names=["a", "b"]), payload[..., 0, :].clone())
    return BaseSolverResult(
        success=torch.tensor([[True, False, True], [False, True, False]], device=device),
        solution=payload.clone(),
        js_solution=state,
        position_error=torch.arange(batch * seeds, device=device, dtype=torch.float32).reshape(batch, seeds),
        rotation_error=torch.ones((batch, seeds), device=device),
        cspace_error=torch.ones((batch, seeds), device=device).mul(2),
        goalset_index=torch.arange(batch * seeds, device=device).reshape(batch, seeds),
        optimized_seeds=payload.add(100),
        metrics=RolloutMetrics(actions=payload.clone(), feasible=torch.ones((batch, seeds), device=device, dtype=torch.bool)),
        seed_rank=torch.tensor([[0, 2, 1], [1, 0, 2]], device=device),
        seed_cost=torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]], device=device),
        total_cost_reshaped=torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]], device=device),
        solution_state=robot,
        feasible=torch.tensor([[True, True, False], [True, True, True]], device=device),
        debug_info={"nested": {"trace": torch.tensor([fill], device=device)}},
        batch_size=batch,
        num_seeds=seeds,
    )


def test_clone_device_move_and_metadata_preserve_tensor_semantics():
    result = _result()
    result.validate()
    assert result.result_batch_size == 2
    assert result.seed_shape == torch.Size((2, 3))

    copied = result.clone()
    copied.solution[0, 0, 0, 0] = -1
    copied.js_solution.velocity[0, 0, 0, 0] = -2
    copied.metrics.actions[0, 0, 0, 0] = -3
    copied.debug_info["nested"]["trace"][0] = -4
    assert result.solution[0, 0, 0, 0] != -1
    assert result.js_solution.velocity[0, 0, 0, 0] != -2
    assert result.metrics.actions[0, 0, 0, 0] != -3
    assert result.debug_info["nested"]["trace"][0] != -4

    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.solution.dtype == torch.float64
    assert moved.js_solution.position.dtype == torch.float64
    assert moved.metrics.actions.dtype == torch.float64
    assert moved.success.dtype == torch.bool
    assert moved.seed_rank.dtype == torch.int64
    assert moved.goalset_index.dtype == torch.int64
    assert moved.solution_state.joint_torque.dtype == torch.float64

    bare = BaseSolverResult(success=torch.ones((3, 2), dtype=torch.bool))
    assert bare.refresh_batch_metadata().batch_size == 3
    assert bare.num_seeds == 2


def test_success_and_batch_merge_include_states_metrics_and_all_seed_channels():
    target = _result(fill=0.0)
    source = _result(fill=1000.0)
    source.success = torch.tensor([[False, True, False], [True, False, False]])
    target.copy_successful_solutions(source)

    assert target.success.tolist() == [[True, True, True], [True, True, False]]
    assert target.solution[0, 1, 0, 0].item() == 1004.0
    assert target.js_solution.jerk[1, 0, 0, 0].item() == 1042.0
    assert target.solution_state.joint_torque[0, 1, 0].item() == 1004.0
    assert target.metrics.actions[1, 0, 0, 0].item() == 1012.0
    # Non-successful entries retain their original target values.
    assert target.solution[0, 0, 0, 0].item() == 0.0

    replacement = _result(fill=2000.0)
    target.copy_at_batch_indices(replacement, torch.tensor([False, True]))
    assert target.solution[1, 2, 0, 0].item() == 2020.0
    assert target.js_solution.acceleration[1, 1, 0, 0].item() == 2036.0
    assert target.metrics.actions[1, 0, 0, 0].item() == 2012.0
    assert target.solution[0, 2, 0, 0].item() == 8.0


def test_merge_rejects_bad_layout_or_device_mismatch_without_partial_copy():
    target, source = _result(), _result()
    source.solution = source.solution[:, :2]
    with pytest.raises(ValueError, match="result tensor shapes differ"):
        target.copy_successful_solutions(source)

    invalid = BaseSolverResult(success=torch.ones((2, 1)))
    with pytest.raises(TypeError, match="torch.bool"):
        invalid.validate()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_base_result_lifecycle_stays_on_mps_without_cpu_fallback():
    target, source = _result("mps"), _result("mps", fill=50.0)
    source.success = torch.tensor([[False, True, False], [False, False, False]], device="mps")
    target.copy_successful_solutions(source)
    copied = target.clone()
    assert copied.device.type == "mps"
    assert copied.solution_state.joint_state.device.type == "mps"
    assert copied.metrics.actions.device.type == "mps"
    assert copied.solution[0, 1, 0, 0].item() == 54.0
