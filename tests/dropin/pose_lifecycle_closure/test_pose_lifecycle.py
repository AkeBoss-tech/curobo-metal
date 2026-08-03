"""Trajectory-prefix and frame-aware lifecycle coverage for portable Pose."""

from __future__ import annotations

import pytest
import torch

from curobo._src.types.pose import Pose


def _trajectory_pose(device: str = "cpu", *, requires_grad: bool = False) -> Pose:
    position = torch.arange(18.0, device=device).reshape(2, 3, 3)
    quaternion = torch.zeros((2, 3, 4), device=device)
    quaternion[..., 0] = 1.0
    if requires_grad:
        position.requires_grad_()
        quaternion.requires_grad_()
    return Pose(position, quaternion, name="camera_frame")


def test_batch_transform_preserves_batch_horizon_point_prefix_and_autograd() -> None:
    pose = _trajectory_pose(requires_grad=True)
    points = torch.arange(36.0).reshape(2, 3, 2, 3).requires_grad_()
    transformed = pose.batch_transform_points(points)
    assert transformed.shape == (2, 3, 2, 3)
    torch.testing.assert_close(transformed, points + pose.position.unsqueeze(-2))
    restored = pose.batch_transform_points_inverse(transformed)
    torch.testing.assert_close(restored, points, atol=1e-6, rtol=0)

    transformed.square().sum().backward()
    assert pose.position.grad is not None and torch.isfinite(pose.position.grad).all()
    assert pose.quaternion.grad is not None and torch.isfinite(pose.quaternion.grad).all()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_repeat_and_seed_expansion_keep_trajectory_and_rotation_representations() -> None:
    pose = _trajectory_pose()
    matrix_backed = Pose.from_matrix(pose.get_matrix())
    matrix_backed.name = pose.name

    repeated = matrix_backed.repeat(2)
    assert repeated.name == "camera_frame"
    assert repeated.position.shape == (4, 3, 3)
    assert repeated.rotation is not None and repeated.rotation.shape == (4, 3, 3, 3)
    torch.testing.assert_close(repeated.position, torch.cat((pose.position, pose.position), dim=0))

    seeded = matrix_backed.repeat_seeds(2)
    assert seeded.position.shape == (4, 3, 3)
    assert seeded.rotation is not None and seeded.rotation.shape == (4, 3, 3, 3)
    torch.testing.assert_close(seeded.position[0], pose.position[0])
    torch.testing.assert_close(seeded.position[1], pose.position[0])
    torch.testing.assert_close(seeded.position[2], pose.position[1])
    torch.testing.assert_close(seeded.position[3], pose.position[1])

    stacked = matrix_backed.stack(matrix_backed.clone())
    assert stacked.position.shape == (4, 3, 3)
    assert stacked.rotation is not None and stacked.rotation.shape == (4, 3, 3, 3)


def test_pose_repeat_count_validation_is_explicit() -> None:
    pose = _trajectory_pose()
    with pytest.raises(TypeError, match="n must be an integer"):
        pose.repeat(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_seeds must be an integer"):
        pose.repeat_seeds("2")  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_trajectory_prefix_transform_and_seed_expansion_stay_on_mps_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    pose = _trajectory_pose("mps", requires_grad=True)
    points = torch.ones((2, 3, 2, 3), device="mps", requires_grad=True)
    output = pose.batch_transform_points(points)
    assert output.device.type == "mps"
    assert pose.repeat_seeds(2).position.device.type == "mps"
    output.square().sum().backward()
    assert points.grad is not None and points.grad.device.type == "mps"
    assert pose.quaternion.grad is not None and pose.quaternion.grad.device.type == "mps"
