"""Lifecycle coverage for the portable pinned MPC solver facade."""

import pytest
import torch
from types import SimpleNamespace

from curobo.model_predictive_control import (
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)


def _solver(*, batch_size: int = 1):
    cfg = ModelPredictiveControlCfg.create(
        "franka.yml",
        max_batch_size=batch_size,
        interpolation_steps=3,
        warm_start_optimization_num_iters=1,
        cold_start_optimization_num_iters=1,
    )
    solver = ModelPredictiveControl(cfg)
    current = solver.default_joint_state
    if batch_size > 1:
        current = current.unsqueeze(0).repeat((batch_size, 1))
    assert solver.setup(current)
    goal = current.clone()
    goal.position = goal.position + 0.01
    assert solver.update_goal_state(goal)
    return solver, current


def test_execution_manager_owns_portable_command_window():
    solver, current = _solver()
    assert solver.trajectory_execution_manager.interpolation_steps == 3

    first = solver.optimize_next_action(current)
    second = solver.optimize_next_action(current)
    third = solver.optimize_next_action(current)
    assert solver.debug_dump()["solve_count"] == 1
    assert [first.metrics["command_index"], second.metrics["command_index"], third.metrics["command_index"]] == [0, 1, 2]
    torch.testing.assert_close(first.next_action.position, first.action_buffer[:, 0])

    fourth = solver.optimize_next_action(current)
    assert solver.debug_dump()["solve_count"] == 2
    assert fourth.metrics["command_index"] == 0
    assert fourth.next_action.position.shape == (1, solver.action_dim)


def test_partial_reset_keeps_other_batch_queue_rows():
    solver, current = _solver(batch_size=2)
    solver.optimize_next_action(current)
    before = solver._action_buffer.clone()

    reset = current.clone()
    reset.position[0] = reset.position[0] + 0.02
    solver.reset_robot_id(reset, torch.tensor([0]))

    assert solver.debug_dump()["command_buffer_valid"]
    torch.testing.assert_close(solver._action_buffer[1], before[1])
    torch.testing.assert_close(solver._action_buffer[0, 0], reset.position[0])
    solve_count = solver.debug_dump()["solve_count"]
    result = solver.optimize_next_action(reset)
    assert solver.debug_dump()["solve_count"] == solve_count
    assert result.next_action.position.shape == (2, solver.action_dim)


def test_seed_update_cannot_reuse_stale_result_metadata_or_empty_manager_state():
    solver, current = _solver()
    first = solver.optimize_next_action(current)
    assert first.metrics["solve_count"] == 1

    seed = current.position.reshape(1, 1, -1).expand(
        -1, solver.action_horizon, -1
    ).clone()
    seed[:, 1:] = seed[:, 1:] + 0.04
    solver.update_seed_trajectory(seed)

    assert solver.trajectory_execution_manager.has_valid_next_command()
    result = solver.optimize_next_action(current)
    assert result.metrics["solve_count"] == 2
    assert result.metrics["command_index"] == 0
    torch.testing.assert_close(result.next_action.position, result.action_buffer[:, 0])


def test_failure_next_action_is_taken_from_safe_fallback_buffer(monkeypatch):
    solver, current = _solver()
    current.velocity = torch.full_like(current.position, 0.2)
    candidate = current.position.reshape(1, 1, -1).expand(
        -1, solver.action_horizon, -1
    ).clone() + 0.8

    def failed_solve(*_args, **_kwargs):
        return SimpleNamespace(
            success=torch.zeros((1, 1), dtype=torch.bool),
            solution=candidate[:, None],
            js_solution=type(current).from_position(candidate[:, None], solver.joint_names),
            solve_time=0.0,
            total_time=0.0,
            feasible=torch.zeros((1, 1), dtype=torch.bool),
            cspace_error=torch.ones((1, 1)),
            optimized_seeds=candidate[:, None],
            seed_cost=torch.ones((1, 1)),
            num_seeds=1,
        )

    monkeypatch.setattr(solver._trajopt, "solve_cspace", failed_solve)
    result = solver.cold_start_solve(current)
    torch.testing.assert_close(result.next_action.position, result.action_buffer[:, 0])
    assert not torch.allclose(result.next_action.position, candidate[:, 0])


def test_cfg_validates_nonterminal_weight_and_preserves_requested_cuda_intent():
    cfg = ModelPredictiveControlCfg.create("franka.yml", use_cuda_graph=True)
    assert cfg.requested_use_cuda_graph and not cfg.use_cuda_graph
    assert cfg.clone(non_terminal_tool_pose_weight_factor=0.0).non_terminal_tool_pose_weight_factor == 0.0
    with pytest.raises(ValueError, match="non_terminal_tool_pose_weight_factor"):
        cfg.clone(non_terminal_tool_pose_weight_factor=-0.1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mpc_command_manager_stays_on_fallback_disabled_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"), torch.float32),
        interpolation_steps=3, warm_start_optimization_num_iters=1,
        cold_start_optimization_num_iters=1,
    )
    solver = ModelPredictiveControl(cfg)
    current = solver.default_joint_state
    assert solver.setup(current)
    goal = current.clone()
    goal.position = goal.position + 0.01
    assert solver.update_goal_state(goal)
    result = solver.optimize_next_action(current)
    assert result.next_action.position.device.type == "mps"
    assert solver.trajectory_execution_manager.get_action_buffer().device.type == "mps"
