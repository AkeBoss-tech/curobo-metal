"""Behavioral coverage for the portable seeded Levenberg--Marquardt solver."""

import pytest
import torch

from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.solver.seed_ik.seed_iteration_state_manager import SeedIterationStateManager
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solver(device_cfg=DeviceCfg("cpu")):
    return SeedIKSolver(SeedIKSolverCfg.create(
        "franka.yml", device_cfg=device_cfg, num_seeds=3, max_iterations=16,
        inner_iterations=4, position_tolerance=0.01, orientation_tolerance=0.1,
        use_cuda_graph=True,
    ))


def _default_goal(solver):
    state = solver.compute_kinematics(solver.default_joint_position)
    pose = state.tool_poses.get_link_pose("panda_hand")
    return GoalToolPose.from_poses({"panda_hand": pose}, ordered_tool_frames=["panda_hand"])


def test_seed_ik_round_trip_and_seed_reset_are_deterministic():
    solver = _solver(device_cfg=DeviceCfg("cpu"))
    goal = _default_goal(solver)
    start = solver.default_joint_position.view(1, 1, -1)
    result = solver.solve_single(goal, seed_config=start)

    assert result.success.tolist() == [[True]]
    torch.testing.assert_close(result.solution[0, 0], start[0, 0], atol=1e-5, rtol=1e-5)
    assert result.metrics["backend"] == "torch-lm"
    assert result.metrics["cuda_graph"] is False

    solver.reset_seed()
    first = solver._generate_seed_configs(1)
    solver.reset_seed()
    torch.testing.assert_close(first, solver._generate_seed_configs(1))


def test_seed_ik_solves_batched_goalset_and_selects_exact_candidate():
    solver = _solver(device_cfg=DeviceCfg("cpu"))
    exact = _default_goal(solver)
    position = exact.position.expand(2, -1, -1, 2, -1).clone()
    quaternion = exact.quaternion.expand(2, -1, -1, 2, -1).clone()
    position[:, :, :, 0, 0] += 0.2
    goal = GoalToolPose(exact.tool_frames, position, quaternion)
    seeds = solver.default_joint_position.repeat(2, 1)

    result = solver.solve_batch(goal, seed_config=seeds, return_seeds=1)
    assert result.success.shape == (2, 1)
    assert result.success.all()
    assert result.goalset_index.tolist() == [[[1]], [[1]]]
    assert result.optimized_seeds.shape == (2, 6, 7)


def test_seed_ik_minibatches_preserve_batched_seed_order_and_velocity_constraints():
    solver = SeedIKSolver(SeedIKSolverCfg.create(
        "franka.yml", num_seeds=2, max_iterations=16, inner_iterations=4,
        position_tolerance=0.01, orientation_tolerance=0.1,
        max_problems_mini_batch=2, velocity_weight=0.1,
        device_cfg=DeviceCfg("cpu"),
    ))
    exact = _default_goal(solver)
    goal = GoalToolPose(
        exact.tool_frames,
        exact.position.expand(3, -1, -1, -1, -1).clone(),
        exact.quaternion.expand(3, -1, -1, -1, -1).clone(),
    )
    current = solver.default_joint_position.expand(3, -1).clone()
    from curobo._src.state.state_joint import JointState
    result = solver.solve_batch(
        goal,
        current_state=JointState(
            current, velocity=torch.zeros_like(current), dt=torch.full((3,), 0.2)
        , device_cfg=DeviceCfg("cpu")),
        seed_config=current,
    )
    assert result.success.shape == (3, 1)
    assert result.success.all()
    assert result.metrics["mini_batch_size"] == 1
    assert result.feasible is not None and result.feasible.all()


def test_seed_ik_rejects_excess_seed_count_and_wrong_goal_dtype():
    solver = _solver(device_cfg=DeviceCfg("cpu"))
    goal = _default_goal(solver)
    with pytest.raises(ValueError, match="more seeds"):
        solver.solve_single(
            goal, seed_config=solver.default_joint_position.view(1, 1, -1).expand(1, 4, -1)
        )
    with pytest.raises(ValueError, match="solver dtype"):
        solver.solve_single(
            GoalToolPose(goal.tool_frames, goal.position.double(), goal.quaternion.double())
        )


def test_seed_error_includes_joint_limit_and_velocity_residuals():
    solver = _solver(device_cfg=DeviceCfg("cpu"))
    goal = _default_goal(solver)
    solver._setup_batch_size(1, 1)
    q = (solver.action_max + 0.2).view(1, -1)
    current = solver.default_joint_position.view(1, -1)
    result = solver.error_calculator.compute_error_and_jacobian(
        q, goal, torch.tensor([0]), current, torch.zeros_like(current), torch.tensor([0.1]), True
    )
    assert result.jacobian.shape[1] == solver.n_residuals
    assert torch.isfinite(result.error_norm).all()
    assert result.error_norm.item() > 0


def test_rejected_iteration_keeps_current_error_not_candidate_error():
    manager = SeedIterationStateManager(
        torch.tensor([-2.0]), torch.tensor([2.0]), 0.5, 2.0, 1e-5, 1e5,
        1e-5, 1e-5, 1.0,
    )
    current = SeedIKState(
        joint_position=torch.tensor([[0.0]]), error_norm=torch.tensor([1.0]),
        jTerror=torch.tensor([[1.0]]), jacobian=torch.ones((1, 1, 1)),
        lambda_damping=torch.ones((1, 1, 1)), position_errors=torch.tensor([1.0]),
        orientation_errors=torch.tensor([1.0]),
    )
    candidate = current.clone()
    candidate.error_norm = torch.tensor([2.0])
    candidate.joint_position = torch.tensor([[1.0]])
    updated = manager.update_iteration_state(current, candidate, torch.tensor([1.0]), 1)
    torch.testing.assert_close(updated.error_norm, current.error_norm)
    torch.testing.assert_close(updated.joint_position, current.joint_position)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_seed_ik_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    solver = _solver(DeviceCfg(torch.device("mps"), torch.float32))
    result = solver.solve_single(
        _default_goal(solver), seed_config=solver.default_joint_position.view(1, 1, -1)
    )
    assert result.success.all()
    assert result.solution.device.type == "mps"
