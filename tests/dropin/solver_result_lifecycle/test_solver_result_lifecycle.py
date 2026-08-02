"""Subclass result lifecycle regressions for portable TrajOpt and MPC."""

from __future__ import annotations

import pytest
import torch

from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


def _joint(position: torch.Tensor, *, dt: torch.Tensor | None = None) -> JointState:
    return JointState(
        position,
        velocity=position + 10,
        acceleration=position + 20,
        jerk=position + 30,
        dt=dt,
        joint_names=["j0", "j1"],
    )


def _traj(device: str = "cpu") -> TrajOptSolverResult:
    payload = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32, device=device).reshape(2, 2, 3, 2)
    return TrajOptSolverResult(
        success=torch.tensor([[True, False], [False, True]], device=device),
        solution=payload.clone(),
        js_solution=_joint(payload.clone(), dt=torch.full((2, 2), 0.25, device=device)),
        interpolated_trajectory=_joint(payload.clone() + 100, dt=torch.full((2, 2), 0.1, device=device)),
        interpolated_last_tstep=torch.full((2, 2), 3, dtype=torch.long, device=device),
        interpolated_metrics=RolloutMetrics(
            feasible=torch.tensor([[True, False], [False, True]], device=device)
        ),
        position_error=torch.zeros((2, 2), device=device),
        rotation_error=torch.zeros((2, 2), device=device),
        seed_cost=torch.tensor([[2.0, 1.0], [3.0, 0.5]], device=device),
        goalset_index=torch.zeros((2, 2, 1), dtype=torch.long, device=device),
        feasible=torch.tensor([[True, False], [False, True]], device=device),
        batch_size=2,
        num_seeds=2,
    )


def _mpc(device: str = "cpu") -> MPCSolverResult:
    values = torch.arange(2 * 3 * 2, dtype=torch.float32, device=device).reshape(2, 3, 2)
    actions = _joint(values.clone(), dt=torch.full((2, 3), 0.2, device=device))
    return MPCSolverResult(
        success=torch.tensor([True, False], device=device),
        solution=values.clone(),
        js_solution=actions.clone(),
        position_error=torch.zeros(2, device=device),
        action_sequence=actions.clone(),
        full_action_sequence=actions.clone(),
        next_action=_joint(values[:, 0].clone(), dt=torch.full((2,), 0.2, device=device)),
        robot_state_sequence=RobotState(actions.clone(), values.clone() + 100),
        action_buffer=values.clone(),
        action_dt=0.2,
        batch_size=2,
    )


def test_trajopt_merges_base_and_interpolated_payloads_and_preserves_rollout_dt() -> None:
    target, source = _traj(), _traj()
    source.solution[1].fill_(91)
    source.js_solution.velocity[1].fill_(92)
    source.interpolated_trajectory.acceleration[1].fill_(93)
    source.interpolated_last_tstep[1].fill_(2)
    source.interpolated_metrics.feasible[1] = True
    target.copy_at_batch_indices(source, torch.tensor([False, True]))

    assert torch.all(target.solution[1] == 91)
    assert torch.all(target.js_solution.velocity[1] == 92)
    assert torch.all(target.interpolated_trajectory.acceleration[1] == 93)
    assert target.interpolated_last_tstep[1].tolist() == [2, 2]
    assert target.interpolated_metrics.feasible[1].tolist() == [True, True]
    # The selected rollout dt, not its allowed maximum, defines real duration.
    target.maximum_trajectory_dt = torch.full((2, 2), 5.0)
    torch.testing.assert_close(target.motion_time(), torch.full((2, 2), 0.5))


def test_trajopt_success_copy_is_seed_granular_and_all_seed_selection_is_identity() -> None:
    target, source = _traj(), _traj()
    source.success[:] = False
    source.success[0, 1] = True
    source.interpolated_trajectory.position[0, 1].fill_(55)
    source.interpolated_last_tstep[0, 1] = 1
    source.interpolated_metrics.feasible[0, 1] = True
    target.copy_successful_solutions(source)

    assert torch.all(target.interpolated_trajectory.position[0, 1] == 55)
    assert target.interpolated_last_tstep[0, 1].item() == 1
    assert target.interpolated_metrics.feasible[0, 1].item()
    assert not torch.all(target.interpolated_trajectory.position[0, 0] == 55)
    target.process_metrics_and_rank_seeds()
    assert target.get_topk_seeds(2) is target


def test_trajopt_to_moves_subclass_fields_without_casting_bool_or_indices() -> None:
    result = _traj().to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert result.solution.dtype == torch.float64
    assert result.interpolated_trajectory.velocity.dtype == torch.float64
    assert result.success.dtype == torch.bool
    assert result.interpolated_last_tstep.dtype == torch.long


def test_mpc_batch_and_success_merges_include_all_action_and_robot_payloads() -> None:
    target, source = _mpc(), _mpc()
    source.action_buffer[0].fill_(44)
    source.action_sequence.velocity[0].fill_(45)
    source.full_action_sequence.acceleration[0].fill_(46)
    source.next_action.jerk[0].fill_(47)
    source.robot_state_sequence.joint_torque[0].fill_(48)
    target.copy_successful_solutions(source)

    assert torch.all(target.action_buffer[0] == 44)
    assert torch.all(target.action_sequence.velocity[0] == 45)
    assert torch.all(target.full_action_sequence.acceleration[0] == 46)
    assert torch.all(target.next_action.jerk[0] == 47)
    assert torch.all(target.robot_state_sequence.joint_torque[0] == 48)
    # The unsuccessful source row does not replace a target command.
    torch.testing.assert_close(target.action_buffer[1], _mpc().action_buffer[1])


def test_mpc_buffer_action_retains_timing_and_layout_checks_plan_alignment() -> None:
    result = _mpc()
    command = result.action_at(1)
    torch.testing.assert_close(command.position, result.action_buffer[:, 1])
    torch.testing.assert_close(command.dt, torch.full((2,), 0.2))
    result.validate_action_layout()

    result.full_action_sequence = _joint(torch.zeros(2, 2, 2))
    with pytest.raises(ValueError, match="full_action_sequence"):
        result.validate_action_layout()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_subclass_result_merges_remain_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    target, source = _traj("mps"), _traj("mps")
    target.copy_successful_solutions(source)
    mpc_target, mpc_source = _mpc("mps"), _mpc("mps")
    mpc_target.copy_successful_solutions(mpc_source)
    assert target.interpolated_trajectory.position.device.type == "mps"
    assert mpc_target.robot_state_sequence.joint_torque.device.type == "mps"
