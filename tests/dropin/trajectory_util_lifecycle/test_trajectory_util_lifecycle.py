"""Portable lifecycle coverage for the pinned trajectory utility modules."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory import (
    TrajInterpolationType,
    get_batch_interpolated_trajectory,
    get_bspline_interpolation,
)
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager


def _spline_state(device: str = "cpu") -> JointState:
    knot = torch.tensor(
        [[[0.0], [1.0], [-1.0], [0.0]]], device=device, requires_grad=True
    )
    state = JointState.from_position(knot.clone())
    state.knot = knot
    state.knot_dt = torch.tensor([0.1], device=device)
    state.control_space = ControlSpace.BSPLINE_3
    return state


def test_cuda_named_bspline_mode_is_a_differentiable_portable_clamped_spline():
    source = _spline_state()
    output, steps = get_batch_interpolated_trajectory(
        source, torch.tensor(0.05), TrajInterpolationType.BSPLINE_KNOTS_CUDA
    )
    assert steps.tolist() == [25]
    assert output.position.shape == (1, 25, 1)
    torch.testing.assert_close(output.position[:, 0], source.knot[:, 0])
    torch.testing.assert_close(output.position[:, -1], source.knot[:, -1])
    assert output.velocity is not None and output.acceleration is not None and output.jerk is not None
    output.position.square().sum().backward()
    assert source.knot.grad is not None


def test_bspline_boundary_tables_variable_lengths_and_output_tail_are_explicit():
    knots = torch.tensor(
        [[[10.0], [1.0], [2.0], [3.0]], [[20.0], [4.0], [5.0], [6.0]]],
        requires_grad=True,
    )
    source = JointState.from_position(knots.clone())
    source.knot = knots
    source.knot_dt = torch.tensor([0.1, 0.2])
    source.control_space = ControlSpace.BSPLINE_3
    out = JointState.zeros((2, 41, 1), DeviceCfg())
    start = JointState.from_position(torch.tensor([[0.0], [100.0]], requires_grad=True))
    goal = JointState.from_position(torch.tensor([[7.0], [200.0]], requires_grad=True))
    result = get_bspline_interpolation(
        source, out, torch.tensor(0.05), current_state=start, goal_state=goal,
        start_idx=torch.tensor([0, 1]), goal_idx=torch.tensor([0, 1]),
        use_implicit_goal_state=torch.tensor([True, False]),
        interpolated_horizon=torch.tensor([25, 41]), bspline_degree=3,
    )
    torch.testing.assert_close(result.position[:, 0, 0], torch.tensor([0.0, 100.0]))
    # The implicit goal selects the first row only; the second preserves its
    # supplied final control point and has no shorter output tail.
    torch.testing.assert_close(result.position[:, -1, 0], torch.tensor([7.0, 6.0]))
    torch.testing.assert_close(result.position[0, 24], result.position[0, -1])
    result.position.sum().backward()
    assert knots.grad is not None and start.position.grad is not None and goal.position.grad is not None


def test_cubic_cuda_parity_path_honors_boundary_table_indices():
    knots = torch.zeros((2, 6, 1), requires_grad=True)
    source = JointState.from_position(knots.clone())
    source.knot = knots
    source.knot_dt = torch.full((2,), 0.2)
    source.control_space = ControlSpace.BSPLINE_3
    start = JointState.from_position(torch.tensor([[10.0], [20.0], [30.0]]))
    goal = JointState.from_position(torch.tensor([[40.0], [50.0], [60.0]]))
    result = get_bspline_interpolation(
        source, JointState.zeros((2, 21, 1), DeviceCfg()), torch.tensor(0.1),
        current_state=start, goal_state=goal,
        start_idx=torch.tensor([2, 0]), goal_idx=torch.tensor([1, 2]),
        use_implicit_goal_state=torch.ones(2, dtype=torch.bool),
        interpolated_horizon=torch.tensor([21, 21]), bspline_degree=3,
    )
    torch.testing.assert_close(result.position[:, 0, 0], torch.tensor([30.0, 10.0]))
    torch.testing.assert_close(result.position[:, -1, 0], torch.tensor([50.0, 60.0]))


def test_spline_mode_rejects_incomplete_or_incompatible_metadata_before_output_mutation():
    source = JointState.from_position(torch.zeros(1, 4, 1))
    with pytest.raises(ValueError, match="knot"):
        get_batch_interpolated_trajectory(
            source, torch.tensor(0.05), TrajInterpolationType.BSPLINE_KNOTS_CUDA
        )
    source.knot = source.position
    source.knot_dt = torch.tensor([0.1])
    source.control_space = ControlSpace.POSITION
    with pytest.raises(ValueError, match="BSPLINE"):
        get_batch_interpolated_trajectory(
            source, torch.tensor(0.05), TrajInterpolationType.BSPLINE_KNOTS_CUDA
        )


def test_execution_manager_offset_window_does_not_overread_short_state_horizon():
    state = JointState.from_position(torch.arange(12.0).reshape(1, 3, 4))
    manager = TrajectoryExecutionManager(9, command_start_idx=2)
    manager.update_state_action_buffers(state, torch.ones(1, 1, 4))
    assert manager.remaining_commands == 1
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 2])
    assert not manager.has_valid_next_command()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_portable_bspline_mode_remains_on_mps_without_cpu_fallback():
    source = _spline_state("mps")
    result, _ = get_batch_interpolated_trajectory(
        source, torch.tensor(0.05, device="mps"), TrajInterpolationType.BSPLINE_KNOTS_CUDA,
        device_cfg=DeviceCfg(torch.device("mps")),
    )
    assert result.position.device.type == "mps"
    result.position.sum().backward()
    assert source.knot.grad is not None
