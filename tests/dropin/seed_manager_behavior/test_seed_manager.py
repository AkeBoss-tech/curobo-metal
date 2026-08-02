"""Pinned-V2 behavioral tests for the portable optimizer seed manager."""

import pytest
import torch

from curobo._src.solver.manager_seed import SeedManager
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _manager(device_cfg=DeviceCfg(), horizon=4):
    return SeedManager(
        device_cfg,
        2,
        torch.tensor([-1.0, -2.0], device=device_cfg.device),
        torch.tensor([1.0, 2.0], device=device_cfg.device),
        random_seed=19,
        action_horizon=horizon,
    )


def test_action_seeds_normalize_layout_pad_and_reset_deterministically():
    manager = _manager()
    # V2 accepts seed-major [N, B, J] action configurations as well as the
    # native batch-major representation.  The one user seed is retained and
    # the remaining two are deterministic Halton samples.
    supplied = torch.tensor([[[0.2, -0.3], [0.4, -0.5]]])
    output = manager.prepare_action_seeds(2, 3, seed_config=supplied)
    assert output.shape == (6, 1, 2)
    torch.testing.assert_close(output.reshape(2, 3, 1, 2)[:, 0, 0], supplied[0])
    assert bool((output[..., 0] >= -1).all())
    assert bool((output[..., 0] <= 1).all())
    manager.reset_seed()
    first = manager.generate_random_actions(2, 4)
    manager.reset_seed()
    torch.testing.assert_close(first, manager.generate_random_actions(2, 4))


def test_trajectory_precedence_interpolation_and_deceleration_layout():
    manager = _manager(horizon=5)
    state = JointState.from_position(
        torch.tensor([[0.0, 0.0], [0.5, -0.5]])
    )
    state.velocity = torch.tensor([[0.2, -0.2], [0.1, 0.1]])
    explicit = torch.full((2, 1, 5, 2), 0.25)
    targets = torch.tensor([[[0.6, 0.8], [0.7, 0.9]], [[-0.1, 0.2], [-0.2, 0.3]]])
    output = manager.prepare_trajectory_seeds(
        2, 3, state, seed_config=targets, seed_traj=explicit
    ).reshape(2, 3, 5, 2)
    torch.testing.assert_close(output[:, 0], explicit[:, 0])
    torch.testing.assert_close(output[:, 1, 0], state.position)
    torch.testing.assert_close(output[:, 1, -1], targets[:, 0])
    deceleration = manager.prepare_deceleration_trajectory_seeds(
        2, 2, state, deceleration_profile="smooth"
    )
    assert deceleration.shape == (4, 5, 2)
    torch.testing.assert_close(
        deceleration.reshape(2, 2, 5, 2)[:, :, 0],
        state.position[:, None].expand(-1, 2, -1),
    )


def test_seed_shapes_and_horizon_boundaries_are_explicit():
    one_step = _manager(horizon=1)
    current = JointState.from_position(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="action_horizon is one"):
        one_step.prepare_trajectory_seeds(1, 1, current)
    with pytest.raises(ValueError, match="Insufficient seed configs"):
        _manager().prepare_trajectory_seeds(
            1, 2, current, seed_config=torch.zeros(1, 1, 2)
        )
    with pytest.raises(ValueError, match="seed_config must have shape"):
        _manager().prepare_action_seeds(1, 1, seed_config=torch.zeros(1, 2, 1))
    assert one_step.generate_random_actions(2, 0).shape == (2, 1, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_seed_manager_mps_is_device_resident_and_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"))
    manager = _manager(device)
    state = JointState.from_position(torch.zeros(1, 2, device="mps"))
    actions = manager.prepare_action_seeds(1, 2)
    trajectories = manager.prepare_trajectory_seeds(1, 2, state)
    assert actions.device.type == "mps"
    assert trajectories.device.type == "mps"
