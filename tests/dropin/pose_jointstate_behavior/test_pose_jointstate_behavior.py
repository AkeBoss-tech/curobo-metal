"""Behavioral checks for the portable V2 pose and joint-state value models."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose


def _z_rotation(angle: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((
        torch.stack((c, -s, zero), dim=-1),
        torch.stack((s, c, zero), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    ), dim=-2)


def test_pose_rotation_constructor_index_equality_and_autograd() -> None:
    angle = torch.tensor([0.2, -0.4], requires_grad=True)
    rotation = _z_rotation(angle)
    position = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], requires_grad=True)
    pose = Pose(position, rotation=rotation, name="camera")

    assert pose.quaternion.shape == (2, 4)
    torch.testing.assert_close(pose.get_rotation(), rotation)
    indexed = pose[1]
    assert indexed.name == "camera"
    assert indexed.position.shape == (1, 3)
    assert indexed.rotation.shape == (1, 3, 3)
    assert pose == pose.clone()
    assert pose != Pose(position + 1e-3, rotation=rotation)

    (pose.quaternion.square().sum() + pose.get_matrix().sum()).backward()
    assert angle.grad is not None and torch.isfinite(angle.grad).all()
    assert position.grad is not None and torch.isfinite(position.grad).all()


def test_joint_state_trajectory_metadata_shape_and_gradient_contract() -> None:
    position = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
    state = JointState(
        position,
        torch.ones_like(position),
        torch.full_like(position, 2.0),
        ["a", "b", "c", "d"],
        torch.full_like(position, 3.0),
        dt=torch.arange(6.0).reshape(2, 3) + 1,
        knot=position * 0.5,
        knot_dt=torch.tensor(0.1),
        aux_data={"source": "test"},
    )

    # dt belongs to the batch/horizon prefix rather than to the DOF tensor.
    reshaped = state.view(2, 3, 4)
    assert reshaped.dt.shape == (2, 3)
    assert reshaped.unsqueeze(0).dt.shape == (2, 3)
    assert reshaped.repeat([1, 1, 1]).knot.shape == (2, 3, 4)

    seeded = state.repeat_seeds(2)
    assert seeded.position.shape == (4, 3, 4)
    assert seeded.dt.shape == (4, 3)
    assert seeded.knot.shape == (4, 3, 4)
    assert seeded.aux_data == {"source": "test"}
    seeded.position.sum().backward()
    torch.testing.assert_close(position.grad, torch.full_like(position, 2.0))


def test_joint_state_shortened_finite_difference_derivatives_remain_indexable() -> None:
    position = torch.arange(8.0).reshape(1, 4, 2)
    state = JointState(
        position,
        velocity=torch.ones(1, 3, 2),
        acceleration=torch.ones(1, 2, 2),
        jerk=torch.ones(1, 1, 2),
        joint_names=["a", "b"],
        dt=torch.full((1, 3), 0.1),
    )
    indexed = state[0]
    assert indexed.position.shape == (4, 2)
    assert indexed.velocity.shape == (3, 2)
    assert indexed.acceleration.shape == (2, 2)
    assert indexed.jerk.shape == (1, 2)
    assert indexed.dt.shape == (1, 3)
    assert indexed.clone().joint_names == ["a", "b"]


def test_pose_and_joint_state_mps_without_cpu_fallback_when_available(monkeypatch) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    angle = torch.tensor([0.3], device="mps", requires_grad=True)
    pose = Pose(torch.zeros(1, 3, device="mps"), rotation=_z_rotation(angle))
    state = JointState.from_position(torch.ones(1, 2, device="mps", requires_grad=True))
    loss = pose.get_matrix().square().sum() + state.repeat_seeds(2).position.sum()
    loss.backward()
    assert pose.position.device.type == "mps"
    assert angle.grad is not None

