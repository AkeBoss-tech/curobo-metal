"""Portable TrajOpt result lifecycle coverage."""

import pytest
import torch

from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState


def _result(device: str = "cpu") -> TrajOptSolverResult:
    batch, seeds, horizon, dof = 2, 3, 5, 2
    solution = torch.arange(batch * seeds * horizon * dof, device=device, dtype=torch.float32)
    solution = solution.reshape(batch, seeds, horizon, dof)
    state = JointState(
        position=solution.clone(),
        velocity=solution.clone() + 100,
        acceleration=solution.clone() + 200,
        jerk=solution.clone() + 300,
        joint_names=["a", "b"],
        dt=torch.full((batch, seeds), 0.1, device=device),
        knot=solution.clone() + 400,
    )
    return TrajOptSolverResult(
        success=torch.tensor([[True, True, False], [True, False, True]], device=device),
        solution=solution,
        js_solution=state,
        interpolated_trajectory=state.clone(),
        interpolated_last_tstep=torch.full((batch, seeds), 4, device=device, dtype=torch.long),
        position_error=torch.arange(batch * seeds, device=device, dtype=torch.float32).reshape(batch, seeds),
        rotation_error=torch.arange(batch * seeds, device=device, dtype=torch.float32).reshape(batch, seeds) + 10,
        cspace_error=torch.arange(batch * seeds, device=device, dtype=torch.float32).reshape(batch, seeds) + 20,
        goalset_index=torch.arange(batch * seeds * 2, device=device).reshape(batch, seeds, 2),
        optimized_seeds=solution.clone() + 500,
        seed_cost=torch.tensor([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]], device=device),
        feasible=torch.tensor([[True, True, False], [True, False, True]], device=device),
        batch_size=batch,
        num_seeds=seeds,
        debug_info={"nested": {"trace": torch.tensor([1.0], device=device)}},
    )


def test_rank_topk_gathers_all_seed_payloads_and_keeps_state_on_device():
    result = _result()
    result.process_metrics_and_rank_seeds()
    selected = result.get_topk_seeds(2)

    # This mirrors the pinned result model: seed_rank retains the selected
    # original rank payload rather than being renumbered after selection.
    assert selected.seed_rank.tolist() == [[0, 1], [1, 2]]
    assert selected.seed_cost.tolist() == [[1.0, 3.0], [1.0, 2.0]]
    assert selected.solution.shape == (2, 2, 5, 2)
    assert selected.goalset_index.shape == (2, 2, 2)
    assert selected.js_solution.position.shape == (2, 2, 5, 2)
    assert selected.interpolated_trajectory.velocity.shape == (2, 2, 5, 2)
    assert selected.interpolated_last_tstep.shape == (2, 2)
    assert selected.num_seeds == 2
    assert selected.metrics is None
    assert selected.device.type == "cpu"


def test_clone_and_success_copy_do_not_alias_joint_or_debug_payloads():
    target = _result()
    source = _result()
    source.solution[0, 1].fill_(999)
    source.js_solution.velocity[0, 1].fill_(333)
    source.interpolated_trajectory.knot[0, 1].fill_(444)
    source.success[:] = False
    source.success[0, 1] = True
    target.copy_successful_solutions(source)
    assert torch.all(target.solution[0, 1] == 999)
    assert torch.all(target.js_solution.velocity[0, 1] == 333)
    assert torch.all(target.interpolated_trajectory.knot[0, 1] == 444)

    copied = target.clone()
    target.js_solution.position[0, 0, 0, 0] = -1
    target.debug_info["nested"]["trace"][0] = -1
    assert copied.js_solution.position[0, 0, 0, 0] != -1
    assert copied.debug_info["nested"]["trace"][0] != -1


def test_interpolated_plan_trim_motion_time_and_batch_copy():
    result = _result()
    plan = result.get_interpolated_plan()
    assert plan.position.shape[-2] == 4
    assert plan.dt.shape == (2, 3)
    assert torch.allclose(result.motion_time(), torch.full((2, 3), 0.4))

    replacement = _result()
    replacement.solution[1].fill_(77)
    result.copy_at_batch_indices(replacement, torch.tensor([False, True]))
    assert torch.all(result.solution[1] == 77)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_result_lifecycle_is_device_resident_on_mps_without_fallback():
    result = _result("mps")
    result.process_metrics_and_rank_seeds()
    selected = result.get_topk_seeds(1)
    assert selected.device.type == "mps"
    assert selected.solution.device.type == "mps"
    assert selected.js_solution.position.device.type == "mps"
    assert selected.motion_time().device.type == "mps"
