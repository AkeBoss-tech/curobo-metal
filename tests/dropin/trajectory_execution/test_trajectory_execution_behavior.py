"""Portable behavioral coverage for trajectory command/seed utility facades."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory import (
    TrajInterpolationType,
    get_batch_interpolated_trajectory,
    get_interpolated_trajectory,
    linear_smooth,
)
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator, interpolate_kernel


def test_execution_lifecycle_preserves_pinned_buffers_and_terminal_action():
    state = JointState.from_position(torch.arange(24.0).reshape(1, 6, 4))
    actions = torch.tensor([[[1.0], [2.0], [3.0]]])
    manager = TrajectoryExecutionManager(2, command_start_idx=1, command_end_idx=4)
    manager.update_state_action_buffers(state, actions)
    assert manager._current_joint_state_trajectory is state
    assert manager._state is state
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 1])
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 2])
    assert not manager.has_valid_next_command()
    shifted = manager.get_shifteaction_dim_buffer()
    torch.testing.assert_close(shifted, torch.tensor([[[2.0], [3.0], [3.0]]]))
    manager.update_action_buffer(actions)
    assert not manager.has_valid_next_command()
    with pytest.raises(ValueError, match="No valid action buffer"):
        manager.get_next_command()


def test_seed_helpers_are_deterministic_differentiable_and_stop_motion():
    cfg = DeviceCfg()
    generator = TrajectorySeedGenerator(6, 2, cfg)
    matrix = interpolate_kernel(3, 4, cfg)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(8))
    assert matrix[0, 0] == 1 and matrix[3, 1] == 1 and matrix[4, 1] == 1

    start = torch.tensor([[0.0, 1.0]], requires_grad=True)
    goal = torch.tensor([[[2.0, -1.0], [1.0, 3.0]]], requires_grad=True)
    seeds = generator.generate_interpolated_seeds(start, goal, 2)
    seeds.square().sum().backward()
    assert start.grad is not None and goal.grad is not None
    torch.testing.assert_close(seeds[:, :, 0], start.detach()[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(seeds[:, :, -1], goal.detach())

    state = JointState.from_position(torch.zeros(1, 2))
    state.velocity = torch.tensor([[1.0, -2.0]])
    state.acceleration = torch.zeros_like(state.velocity)
    state.dt = torch.tensor([0.1])
    decelerated = generator.generate_deceleration_seeds(state, 2, deceleration_profile="smooth")
    torch.testing.assert_close(decelerated[:, :, 0], state.position[:, None].expand(-1, 2, -1))
    displacement = decelerated[:, 0, 1:] - decelerated[:, 0, :-1]
    assert torch.all(displacement[..., 0] >= -1e-6)
    assert torch.all(displacement[..., 1] <= 1e-6)


def test_cubic_quintic_and_numpy_helper_are_meaningful_and_keep_endpoints():
    raw = JointState.from_position(torch.tensor([[[0.0], [1.0], [-1.0], [0.0]]], requires_grad=True))
    raw.dt = torch.tensor([0.1])
    linear, _ = get_batch_interpolated_trajectory(raw, torch.tensor(0.04), TrajInterpolationType.LINEAR)
    cubic, steps = get_batch_interpolated_trajectory(raw, torch.tensor(0.04), TrajInterpolationType.CUBIC)
    quintic, _ = get_batch_interpolated_trajectory(raw, torch.tensor(0.04), TrajInterpolationType.QUINTIC)
    assert steps.tolist() == [10]
    torch.testing.assert_close(cubic.position[:, 0], raw.position[:, 0])
    torch.testing.assert_close(cubic.position[:, -1], raw.position[:, -1])
    assert not torch.allclose(cubic.position, linear.position)
    assert not torch.allclose(quintic.position, linear.position)
    cubic.position.sum().backward()
    assert raw.position.grad is not None
    numpy_cubic = linear_smooth([0.0, 1.0, -1.0, 0.0], n=10, kind=TrajInterpolationType.CUBIC)
    assert numpy_cubic.shape == (10,)
    assert numpy_cubic[0] == 0 and numpy_cubic[-1] == 0


def test_list_interpolation_returns_pinned_result_metadata():
    output = JointState.zeros((2, 8, 1), DeviceCfg())
    result, steps, dt = get_interpolated_trajectory(
        [torch.tensor([[0.0], [1.0]]), torch.tensor([[2.0], [4.0]])],
        output,
        des_horizon=5,
        interpolation_dt=0.1,
        kind=TrajInterpolationType.LINEAR,
    )
    assert result.position.shape == (2, 8, 1)
    assert steps == [5, 5]
    torch.testing.assert_close(dt, torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(result.position[:, 4, 0], torch.tensor([1.0, 4.0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_interpolation_stays_on_device_without_fallback():
    cfg = DeviceCfg(torch.device("mps"))
    raw = JointState.from_position(torch.tensor([[[0.0], [1.0], [0.0]]], device="mps"))
    raw.dt = torch.tensor([0.1], device="mps")
    output, _ = get_batch_interpolated_trajectory(raw, torch.tensor(0.05, device="mps"), TrajInterpolationType.CUBIC, device_cfg=cfg)
    assert output.position.device.type == "mps"
