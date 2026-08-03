"""Focused portable lifecycle coverage for goal, seed, and seeded-IK managers."""

import pytest
import torch

from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.solver.seed_ik.seed_iteration_state_manager import SeedIterationStateManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solve(batch_size=2, goalset=1, seeds=2):
    return SolveState(SolveMode.BATCH, batch_size, 1, num_goalset=goalset, num_seeds=seeds)


def _pose(batch_size=2, goalset=1, device="cpu"):
    return GoalToolPose(
        ["tool"],
        torch.zeros(batch_size, 1, 1, goalset, 3, device=device),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(batch_size, 1, 1, goalset, 1),
    )


def _iteration_state(device="cpu", position=0.0, error=0.1):
    return SeedIKState(
        joint_position=torch.tensor([[position]], device=device),
        error_norm=torch.tensor([error], device=device),
        jTerror=torch.ones(1, 1, device=device),
        jacobian=torch.ones(1, 1, 1, device=device),
        lambda_damping=torch.ones(1, 1, 1, device=device),
        position_errors=torch.tensor([0.0], device=device),
        orientation_errors=torch.tensor([0.0], device=device),
    )


def _iteration_manager(*, joint_limit_weight=1.0, device="cpu"):
    return SeedIterationStateManager(
        torch.tensor([-1.0], device=device), torch.tensor([1.0], device=device),
        0.1, 2.0, 1.0e-5, 1.0e3, 0.01, 0.02, joint_limit_weight,
    )


def test_goal_manager_rejects_malformed_goal_and_state_payloads_before_buffer_creation():
    manager = GoalManager(DeviceCfg())
    with pytest.raises(ValueError, match="goalset size"):
        manager.update_goal_buffer(_solve(goalset=2), goal_tool_poses=_pose(goalset=1))
    with pytest.raises(ValueError, match="batch size"):
        manager.update_goal_buffer(_solve(), current_js=JointState.from_position(torch.zeros(1, 2)))
    with pytest.raises(ValueError, match="current_state_dt"):
        manager.update_goal_buffer(_solve(), current_state_dt=torch.ones(3))


def test_goal_manager_accepts_compact_single_problem_joint_state():
    manager = GoalManager(DeviceCfg())
    solve_state = SolveState(SolveMode.BATCH, 1, 1, num_goalset=1, num_seeds=2)
    current = JointState.from_position(torch.zeros(2))
    registry, update_reference = manager.update_goal_buffer(solve_state, current_js=current)
    assert update_reference
    assert registry.current_js is current


def test_seed_manager_converts_all_state_channels_together_and_rejects_nonfinite_seeds():
    manager = SeedManager(
        DeviceCfg(), 2, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]), action_horizon=4
    )
    state = JointState.from_position(torch.zeros(1, 2, dtype=torch.float64))
    state.velocity = torch.ones(1, 2, dtype=torch.float64)
    state.acceleration = torch.zeros(1, 2, dtype=torch.float64)
    state.dt = torch.ones(1, dtype=torch.float64) * 0.1
    output = manager.prepare_deceleration_trajectory_seeds(1, 1, state)
    assert output.dtype == torch.float32
    with pytest.raises(ValueError, match="finite"):
        manager.prepare_action_seeds(1, 1, seed_config=torch.tensor([[[float("nan"), 0.0]]]))


def test_iteration_manager_honors_joint_limit_mode_and_preserves_rejected_state():
    current = _iteration_state(position=0.0, error=0.2)
    candidate = _iteration_state(position=1.0, error=0.3)
    strict = _iteration_manager()
    updated = strict.update_iteration_state(current, candidate, torch.tensor([0.1]), 1)
    assert not updated.improvement.item()
    torch.testing.assert_close(updated.error_norm, current.error_norm)
    assert strict.convergence_position_tolerance == strict.position_tolerance
    # A valid pose at an exact boundary is rejected only when joint-limit
    # convergence is enabled; V2 exposes this as a configurable weight.
    assert not strict._check_convergence(candidate.joint_position, candidate.position_errors, candidate.orientation_errors).item()
    relaxed = _iteration_manager(joint_limit_weight=0.0)
    assert relaxed._check_convergence(candidate.joint_position, candidate.position_errors, candidate.orientation_errors).item()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_goal_seed_and_iteration_managers_remain_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"))
    goal = GoalManager(device).create_goal_buffer(_solve(1), goal_tool_poses=_pose(1, device="mps"))
    seeds = SeedManager(
        device, 1, torch.tensor([-1.0], device="mps"), torch.tensor([1.0], device="mps"), action_horizon=2
    ).prepare_trajectory_seeds(1, 2, JointState.from_position(torch.zeros(1, 1, device="mps")))
    current, candidate = _iteration_state("mps", 0.0, 1.0), _iteration_state("mps", 0.1, 0.2)
    updated = _iteration_manager(device="mps").update_iteration_state(current, candidate, torch.ones(1, device="mps"), 1)
    assert goal.idxs_link_pose.device.type == seeds.device.type == updated.joint_position.device.type == "mps"
