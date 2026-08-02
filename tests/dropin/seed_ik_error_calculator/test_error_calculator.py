"""Direct lifecycle and residual coverage for the portable SeedIK calculator."""

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solver(device_cfg=DeviceCfg(), **kwargs):
    return SeedIKSolver(
        SeedIKSolverCfg.create(
            "franka.yml",
            device_cfg=device_cfg,
            num_seeds=1,
            max_iterations=4,
            inner_iterations=2,
            **kwargs,
        )
    )


def _goal_at_default(solver):
    state = solver.compute_kinematics(solver.default_joint_position)
    pose = state.tool_poses.get_link_pose("panda_hand")
    return GoalToolPose.from_poses({"panda_hand": pose}, ordered_tool_frames=["panda_hand"])


def test_error_calculator_criteria_control_residual_axes_and_workspace_lifecycle():
    solver = _solver()
    calculator = solver.error_calculator
    calculator.setup_batch_tensors(1)
    first_workspace = calculator._cost_shape
    calculator.setup_batch_tensors(1)
    assert calculator._cost_shape is first_workspace

    goal = _goal_at_default(solver)
    y_only_offset = GoalToolPose(
        goal.tool_frames,
        goal.position + torch.tensor([0.0, 0.2, 0.0]).view(1, 1, 1, 1, 3),
        goal.quaternion.clone(),
    )
    q = solver.default_joint_position.view(1, -1)
    calculator.update_tool_pose_criteria(
        {"panda_hand": ToolPoseCriteria.track_position([1.0, 0.0, 0.0])}
    )
    masked = calculator.compute_error_and_jacobian(q, y_only_offset, torch.tensor([0]))
    torch.testing.assert_close(masked.position_errors, torch.zeros_like(masked.position_errors))
    torch.testing.assert_close(masked.jTerror, torch.zeros_like(masked.jTerror), atol=1e-6, rtol=0)

    calculator.update_tool_pose_criteria({"panda_hand": ToolPoseCriteria.disabled()})
    disabled = calculator.compute_error_and_jacobian(q, goal, torch.tensor([0]))
    torch.testing.assert_close(disabled.jacobian[:, :6], torch.zeros_like(disabled.jacobian[:, :6]))
    torch.testing.assert_close(disabled.error_norm, torch.zeros_like(disabled.error_norm), atol=1e-6, rtol=0)


def test_error_calculator_omits_disabled_limit_rows_and_has_physical_accel_gradient():
    solver = _solver(joint_limit_weight=0.0, velocity_weight=4.0, acceleration_weight=9.0)
    solver._setup_batch_size(1, 1)
    goal = _goal_at_default(solver)
    q = solver.default_joint_position.view(1, -1)
    result = solver.error_calculator.compute_error_and_jacobian(
        q,
        goal,
        torch.tensor([0]),
        current_position=q.clone(),
        current_velocity=torch.full_like(q, 0.1),
        dt=torch.tensor([0.5]),
    )
    # Six pose components plus one velocity and acceleration residual per DOF;
    # no zero-valued joint-limit block is emitted when it is disabled.
    assert result.jacobian.shape == (1, 6 + 2 * solver.dof, solver.dof)
    torch.testing.assert_close(result.jTerror, torch.full_like(q, -1.8), atol=2e-5, rtol=0)
    torch.testing.assert_close(result.error_norm, torch.tensor([0.63]), atol=2e-5, rtol=0)


def test_error_calculator_rejects_nonfinite_goal_and_wrong_problem_batch():
    solver = _solver()
    solver._setup_batch_size(1, 1)
    goal = _goal_at_default(solver)
    bad_goal = GoalToolPose(goal.tool_frames, torch.full_like(goal.position, float("nan")), goal.quaternion)
    with pytest.raises(ValueError, match="finite"):
        solver.error_calculator.compute_error_and_jacobian(
            solver.default_joint_position.view(1, -1), bad_goal, torch.tensor([0])
        )
    with pytest.raises(ValueError, match="num_problems"):
        solver.error_calculator.compute_error_and_jacobian(
            solver.default_joint_position.repeat(2, 1), goal, torch.tensor([0, 0])
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_error_calculator_mps_criteria_and_acceleration_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    solver = _solver(
        DeviceCfg(torch.device("mps"), torch.float32),
        velocity_weight=0.1,
        acceleration_weight=0.1,
    )
    solver._setup_batch_size(1, 1)
    calculator = solver.error_calculator
    calculator.update_tool_pose_criteria(
        {"panda_hand": ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1.0] * 6,
            device_cfg=solver.device_cfg,
        )}
    )
    q = solver.default_joint_position.view(1, -1)
    result = calculator.compute_error_and_jacobian(
        q, _goal_at_default(solver), torch.tensor([0], device="mps"), q,
        torch.zeros_like(q), torch.tensor([0.1], device="mps"), True,
    )
    assert result.jacobian.device.type == "mps"
    assert result.jTerror.device.type == "mps"
    assert torch.isfinite(result.error_norm).all()
