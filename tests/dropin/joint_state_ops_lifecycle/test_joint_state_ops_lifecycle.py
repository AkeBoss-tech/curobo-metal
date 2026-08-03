"""CPU/MPS lifecycle coverage for portable JointState operation modules."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import (
    apply_kernel_to_joint_state,
    augment_joint_state,
    blend_joint_states,
    calculate_fd_from_position,
    cat_joint_states,
    repeat_joint_state,
    scale_joint_state_by_dt,
)
from curobo._src.state.state_joint_trajectory_ops import (
    copy_joint_state_at_batch_seed_indices,
    copy_joint_state_at_index,
    gather_joint_state_by_seed,
    get_joint_state_at_horizon_index,
    index_joint_state_dof,
    trim_joint_state_trajectory,
)


def _state(device: str = "cpu") -> JointState:
    position = torch.arange(2 * 3 * 4 * 2.0, device=device).reshape(2, 3, 4, 2).requires_grad_()
    return JointState(
        position, position + 1.0, position + 2.0, ["j0", "j1"], position + 3.0,
        dt=torch.arange(6.0, device=device).reshape(2, 3) + 0.1,
        knot=position[..., :3, :] if False else torch.ones(2, 3, 2, 2, device=device),
        knot_dt=torch.full((2, 3), 0.2, device=device),
    )


def test_ops_keep_metadata_and_autograd() -> None:
    state = _state()
    repeated = repeat_joint_state(state, [2, 1, 1, 1])
    assert repeated.position.shape == (4, 3, 4, 2)
    assert repeated.dt.shape == (4, 3)
    assert repeated.knot.shape == (4, 3, 2, 2)

    kernel = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    applied = apply_kernel_to_joint_state(state, kernel)
    assert applied.position.shape == (2, 3, 4, 2)
    torch.testing.assert_close(applied.dt[1], (state.dt[0] + state.dt[1]) / 2)

    rescaled = scale_joint_state_by_dt(state, torch.tensor(0.2), torch.tensor(0.1))
    torch.testing.assert_close(rescaled.velocity, state.velocity * 2)
    torch.testing.assert_close(rescaled.knot_dt, state.knot_dt * 0.5)
    (applied.position.square().sum() + rescaled.velocity.sum()).backward()
    assert state.position.grad is not None


def test_blend_cat_augment_and_finite_difference_boundaries() -> None:
    target = JointState.from_position(torch.zeros(1, 2), ["a", "b"])
    source = JointState.from_position(torch.ones(1, 2), ["a", "b"])
    blend_joint_states(target, source, FilterCoeff(position=0.25))
    torch.testing.assert_close(target.position, torch.full((1, 2), 0.25))

    appended = cat_joint_states(target, source, -1)
    assert appended.joint_names == ["a", "b", "a", "b"]
    augmented = augment_joint_state(
        JointState.from_position(torch.tensor([[1.0]]), ["a"]), ["a", "b"],
        JointState.from_position(torch.tensor([[2.0]]), ["b"]),
    )
    torch.testing.assert_close(augmented.position, torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="duplicates"):
        augment_joint_state(target, ["a", "a"])

    trajectory = JointState.from_position(torch.arange(12.0).reshape(1, 6, 2))
    calculate_fd_from_position(trajectory, 0.1)
    assert trajectory.velocity.shape[-2] == 5
    assert trajectory.acceleration.shape[-2] == 4


def test_batched_seed_copy_gather_trim_and_dof_index() -> None:
    state = _state()
    selected = gather_joint_state_by_seed(state, torch.tensor([[2, 0], [1, 1]]))
    assert selected.position.shape == (2, 2, 4, 2)
    torch.testing.assert_close(selected.position[0, 0], state.position[0, 2])
    torch.testing.assert_close(selected.dt[1, 1], state.dt[1, 1])

    target = _state()
    source = _state()
    source.position = source.position + 1000
    copy_joint_state_at_batch_seed_indices(target, source, torch.tensor([0]), torch.tensor([2]))
    torch.testing.assert_close(target.position[0, 2], source.position[0, 2])
    copy_joint_state_at_index(target, source[0], 1)
    torch.testing.assert_close(target.position[1], source.position[0])

    waypoint = get_joint_state_at_horizon_index(state, 2)
    assert waypoint.position.shape == (2, 3, 2)
    trimmed = trim_joint_state_trajectory(state, 1, 3)
    assert trimmed.position.shape == (2, 3, 2, 2)
    assert trimmed.knot is None and trimmed.knot_dt is None
    indexed = index_joint_state_dof(state, torch.tensor([1, 0]))
    assert indexed.joint_names == ["j1", "j0"]
    torch.testing.assert_close(indexed.dt, state.dt)


def test_mps_ops_without_cpu_fallback_when_available(monkeypatch) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    state = _state("mps")
    result = gather_joint_state_by_seed(state, torch.tensor([[0], [2]], device="mps"))
    result = index_joint_state_dof(result, torch.tensor([1], device="mps"))
    (result.position.square().sum()).backward()
    assert result.position.device.type == "mps"
    assert state.position.grad is not None and state.position.grad.device.type == "mps"
