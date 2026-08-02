"""Portable lifecycle coverage for the pinned IK solver result model."""

import pytest
import torch

from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _result(device="cpu") -> IKSolverResult:
    solution = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    return IKSolverResult(
        success=torch.tensor([[True, False, True], [False, True, True]], device=device),
        solution=solution,
        js_solution=JointState(
            solution.clone(),
            velocity=solution.add(100),
            acceleration=solution.add(200),
            jerk=solution.add(300),
            dt=torch.full((2, 3), 0.1, device=device),
            joint_names=["a", "b", "c", "d"],
        ),
        position_error=torch.tensor([[0.3, 0.1, 0.2], [0.4, 0.2, 0.3]], device=device),
        seed_cost=torch.tensor([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]], device=device),
        seed_rank=torch.tensor([[2, 0, 1], [1, 2, 0]], device=device),
        goalset_index=torch.arange(6, device=device).reshape(2, 3, 1),
        feasible=torch.tensor([[True, True, False], [True, False, True]], device=device),
        optimized_seeds=solution.add(400),
        metrics={"per_seed": solution[..., 0].clone(), "nested": {"value": solution[..., 1].clone()}},
        batch_size=2,
        num_seeds=3,
    )


def test_rank_and_seed_selection_preserve_all_seed_payloads():
    result = _result()
    selected = result.get_topk_seeds(2)

    assert selected.seed_cost.tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert selected.solution.shape == (2, 2, 4)
    assert selected.js_solution.position.shape == (2, 2, 4)
    assert selected.js_solution.velocity[0, 0, 0].item() == 104.0
    assert selected.metrics["per_seed"].tolist() == [[4.0, 8.0], [20.0, 12.0]]
    assert selected.num_seeds == 2

    best = result.best_seed()
    assert best.solution.shape == (2, 1, 4)
    assert best.seed_cost.tolist() == [[1.0], [1.0]]
    assert best.seed_rank.tolist() == [[0], [0]]


def test_clone_batch_select_and_feasibility_lifecycle_are_isolated():
    result = _result()
    copied = result.clone()
    copied.solution[0, 0, 0] = -1
    copied.metrics["nested"]["value"][0, 0] = -1
    copied.js_solution.velocity[0, 0, 0] = -1
    assert result.solution[0, 0, 0].item() == 0.0
    assert result.metrics["nested"]["value"][0, 0].item() == 1.0
    assert result.js_solution.velocity[0, 0, 0].item() == 100.0

    one = result.select_batch(1)
    assert one.batch_size == 1
    assert one.solution.shape == (1, 3, 4)
    assert one.js_solution.dt.shape == (1, 3)

    result.process_metrics_and_rank_seeds()
    assert result.seed_rank.tolist() == [[1, 2, 0], [2, 0, 1]]
    assert result.success.tolist() == [[True, False, False], [False, False, True]]


def test_copy_successful_solutions_preserves_joint_derivative_channels():
    target = _result()
    source = _result()
    source.success = torch.tensor([[False, True, False], [True, False, False]])
    source.js_solution.velocity.fill_(77)
    source.js_solution.acceleration.fill_(88)
    source.js_solution.jerk.fill_(99)
    target.copy_successful_solutions(source)

    assert target.success.tolist() == [[True, True, True], [True, True, True]]
    assert target.js_solution.velocity[0, 1, 0].item() == 77
    assert target.js_solution.acceleration[1, 0, 0].item() == 88
    assert target.js_solution.jerk[1, 0, 0].item() == 99


def test_result_to_preserves_bool_and_integer_metadata():
    result = _result()
    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.solution.dtype == torch.float64
    assert moved.success.dtype == torch.bool
    assert moved.seed_rank.dtype == torch.int64
    assert moved.goalset_index.dtype == torch.int64
    assert moved.js_solution.position.dtype == torch.float64
    assert moved.device == torch.device("cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_result_lifecycle_stays_on_mps_without_cpu_fallback():
    result = _result("mps")
    selected = result.get_topk_seeds(2)
    selected.process_metrics_and_rank_seeds()
    assert selected.device.type == "mps"
    assert selected.solution.device.type == "mps"
    assert selected.js_solution.velocity.device.type == "mps"
    assert selected.seed_rank.device.type == "mps"
