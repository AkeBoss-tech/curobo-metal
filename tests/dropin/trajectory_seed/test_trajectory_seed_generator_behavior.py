"""CPU/MPS behavioral coverage for portable V2 trajectory seeds."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator


def test_singleton_start_broadcasts_across_goal_batches_and_preserves_autograd():
    generator = TrajectorySeedGenerator(6, 2, DeviceCfg())
    start = torch.tensor([[0.0, 1.0]], requires_grad=True)
    goals = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0]], [[-1.0, 0.0], [3.0, -2.0]]],
        requires_grad=True,
    )
    seeds = generator.generate_interpolated_seeds(start, goals, num_seeds=2)
    assert seeds.shape == (2, 2, 6, 2)
    torch.testing.assert_close(seeds[:, :, 0], start.detach().expand(2, 2, 2))
    torch.testing.assert_close(seeds[:, :, -1], goals.detach())
    seeds.square().sum().backward()
    assert start.grad is not None and goals.grad is not None


def test_deceleration_time_uses_active_prefix_and_stops_per_batch_dt():
    generator = TrajectorySeedGenerator(8, 1, DeviceCfg())
    state = JointState(
        position=torch.tensor([[0.0], [2.0]]),
        velocity=torch.tensor([[2.0], [-3.0]]),
        acceleration=torch.tensor([[0.5], [-0.25]]),
        dt=torch.tensor([[0.1], [0.2]]),
    )
    seeds = generator.generate_deceleration_seeds(
        state, 3, deceleration_time=torch.tensor([0.25, 0.45]), deceleration_profile="smooth"
    )
    assert seeds.shape == (2, 3, 8, 1)
    torch.testing.assert_close(seeds[:, 0], seeds[:, 1])
    # Integration never reverses; terminal knots are stationary for different
    # per-batch dt values.
    delta = seeds[:, 0, 1:] - seeds[:, 0, :-1]
    assert torch.all(delta[0] >= -1e-6)
    assert torch.all(delta[1] <= 1e-6)
    torch.testing.assert_close(delta[:, -1], torch.zeros_like(delta[:, -1]), atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_seed_modes_remain_on_mps_without_fallback():
    cfg = DeviceCfg(torch.device("mps"))
    generator = TrajectorySeedGenerator(5, 2, cfg)
    start = torch.tensor([[0.0, 1.0]], device="mps", requires_grad=True)
    goal = torch.tensor([[[1.0, -1.0]]], device="mps", requires_grad=True)
    interpolated = generator.generate_interpolated_seeds(start, goal, 1)
    state = JointState(
        position=start.detach(),
        velocity=torch.tensor([[1.0, -1.0]], device="mps"),
        acceleration=torch.zeros(1, 2, device="mps"),
        dt=torch.tensor([0.05], device="mps"),
    )
    deceleration = generator.generate_deceleration_seeds(state, 2, deceleration_time=0.1)
    assert interpolated.device.type == "mps"
    assert deceleration.device.type == "mps"
    interpolated.sum().backward()
    assert start.grad is not None and goal.grad is not None
