"""Closure coverage for pinned JointState composition lifecycle semantics."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.state_joint import JointState


def test_stack_concatenates_trajectory_axis_and_preserves_autograd() -> None:
    left_position = torch.arange(12.0).reshape(2, 2, 3).requires_grad_()
    right_position = (torch.arange(12.0).reshape(2, 2, 3) + 20).requires_grad_()
    left = JointState.from_position(left_position, ["a", "b", "c"])
    right = JointState.from_position(right_position, ["a", "b", "c"])

    result = left.stack(right)
    assert result.position.shape == (2, 4, 3)
    torch.testing.assert_close(result.position[:, :2], left_position)
    torch.testing.assert_close(result.position[:, 2:], right_position)
    result.position.square().sum().backward()
    assert left_position.grad is not None
    assert right_position.grad is not None


def test_append_broadcasts_locked_joints_and_keeps_explicit_derivatives() -> None:
    active_position = torch.arange(2 * 3 * 2.0).reshape(2, 3, 2).requires_grad_()
    active = JointState(
        active_position,
        torch.ones_like(active_position),
        None,
        ["a", "b"],
        torch.full_like(active_position, 4.0),
        dt=torch.tensor([0.1, 0.2]),
    )
    locked_position = torch.tensor([10.0, 20.0], requires_grad=True)
    locked = JointState(
        locked_position,
        torch.tensor([3.0, 5.0]),
        None,
        ["c", "d"],
        dt=torch.tensor([0.05]),
    )

    result = active.append_joints(locked)
    assert result.position.shape == (2, 3, 4)
    assert result.joint_names == ["a", "b", "c", "d"]
    torch.testing.assert_close(result.position[..., 2:], locked_position.expand(2, 3, 2))
    torch.testing.assert_close(result.velocity[..., 2:], locked.velocity.expand(2, 3, 2))
    torch.testing.assert_close(result.jerk[..., 2:], torch.zeros(2, 3, 2))
    torch.testing.assert_close(result.dt, locked.dt)
    result.position.sum().backward()
    torch.testing.assert_close(locked_position.grad, torch.full((2,), 6.0))


def test_append_rejects_non_broadcastable_and_shared_knot_layouts() -> None:
    active = JointState.from_position(torch.zeros(2, 3, 1), ["a"])
    incompatible = JointState.from_position(torch.zeros(4, 1), ["b"])
    with pytest.raises(ValueError, match="appending joints requires"):
        active.append_joints(incompatible)

    left = JointState.from_position(torch.zeros(1, 1), ["a"])
    left.knot = torch.zeros(1, 2, 1)
    right = JointState.from_position(torch.zeros(1, 1), ["b"])
    right.knot = torch.zeros(1, 2, 1)
    with pytest.raises(NotImplementedError, match="knot append"):
        left.append_joints(right)


def test_copy_data_reuses_or_adopts_channel_allocation() -> None:
    target = JointState.from_position(torch.zeros(1, 2), ["a", "b"])
    source = JointState.from_position(torch.ones(3, 2), ["a", "b"])
    original = target.position.data_ptr()
    target.copy_data(source)
    assert target.position.data_ptr() == source.position.data_ptr()
    assert target.position.data_ptr() != original

    reusable = JointState.from_position(torch.zeros(3, 2), ["a", "b"])
    reusable_pointer = reusable.position.data_ptr()
    reusable.copy_data(source)
    assert reusable.position.data_ptr() == reusable_pointer
    torch.testing.assert_close(reusable.position, source.position)
