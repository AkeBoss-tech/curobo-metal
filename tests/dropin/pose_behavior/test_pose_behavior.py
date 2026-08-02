"""Pinned V2 Pose raw-helper and value-lifecycle coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.types.pose import (
    Pose,
    batch_transform_points,
    batch_transform_points_inverse,
    pose_inverse,
    pose_multiply,
    pose_to_affine_matrix,
    pose_to_matrix,
    transform_points,
)


def _quarter_turn_pose(*, device: str = "cpu") -> Pose:
    half_sqrt = 2.0**-0.5
    return Pose(
        torch.tensor([[1.0, -2.0, 0.5]], device=device),
        torch.tensor([[half_sqrt, 0.0, 0.0, half_sqrt]], device=device),
        name="camera",
    )


def test_raw_tensor_pose_helpers_match_pinned_signatures_and_output_buffers() -> None:
    pose = _quarter_turn_pose()
    position, quaternion = pose.position.requires_grad_(), pose.quaternion.requires_grad_()
    point = torch.tensor([[[2.0, 0.0, 1.0]]])

    matrix_out = torch.empty((1, 4, 4))
    affine_out = torch.empty((1, 3, 4))
    assert pose_to_matrix(position, quaternion, matrix_out) is matrix_out
    assert pose_to_affine_matrix(position, quaternion, affine_out) is affine_out
    torch.testing.assert_close(matrix_out[..., :3, :], affine_out)

    point_out = torch.empty_like(point)
    transformed = transform_points(position, quaternion, point, point_out)
    assert transformed is point_out
    round_trip = batch_transform_points_inverse(position, quaternion, transformed)
    torch.testing.assert_close(round_trip, point, atol=1e-6, rtol=0)

    inverse_position_out, inverse_quaternion_out = torch.empty_like(position), torch.empty_like(quaternion)
    inverse_position, inverse_quaternion = pose_inverse(
        position, quaternion, inverse_position_out, inverse_quaternion_out
    )
    assert inverse_position is inverse_position_out
    assert inverse_quaternion is inverse_quaternion_out
    zero, identity = pose_multiply(position, quaternion, inverse_position, inverse_quaternion)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-6, rtol=0)
    torch.testing.assert_close(identity, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    (matrix_out.square().sum() + transformed.square().sum()).backward()
    assert position.grad is not None and torch.isfinite(position.grad).all()
    assert quaternion.grad is not None and torch.isfinite(quaternion.grad).all()


def test_pose_value_lifecycle_preserves_matrix_representation_and_name() -> None:
    base = _quarter_turn_pose()
    matrix_pose = Pose.from_matrix(base.get_matrix())
    matrix_pose.name = "matrix_camera"
    repeated = matrix_pose.repeat_seeds(3)
    assert repeated.name == "matrix_camera"
    assert repeated.rotation is not None and repeated.rotation.shape == (3, 3, 3)
    torch.testing.assert_close(repeated.rotation, matrix_pose.rotation.expand(3, -1, -1))

    stacked = matrix_pose.stack(matrix_pose.clone())
    assert stacked.name == "matrix_camera"
    assert stacked.rotation is not None and stacked.rotation.shape == (2, 3, 3)
    concatenated = Pose.cat([matrix_pose, matrix_pose.clone()])
    assert concatenated.name == "matrix_camera"
    assert concatenated.rotation is not None

    destination = matrix_pose.clone()
    source = Pose.from_matrix(_quarter_turn_pose().get_matrix())
    source.rotation[..., 0, 0] = 0.9
    destination.copy_(source)
    torch.testing.assert_close(destination.rotation, source.rotation)
    assert destination.contiguous().name == "matrix_camera"


def test_batch_pose_helpers_accept_noncontiguous_inputs_and_pose_convenience() -> None:
    pose = _quarter_turn_pose()
    points = torch.arange(12.0).reshape(1, 2, 2, 3).transpose(1, 2)
    # Pose's legacy point method deliberately flattens high-rank input, while
    # the raw batch helper retains the [batch, ... , 3] layout.
    flattened = pose.transform_points(points)
    assert flattened.shape == (4, 3)
    batched = batch_transform_points(pose.position, pose.quaternion, points.reshape(1, 4, 3))
    torch.testing.assert_close(flattened, batched.reshape(-1, 3))

    output = torch.empty_like(flattened)
    assert transform_points(pose, points, output) is output
    torch.testing.assert_close(output, flattened)


def test_pose_helpers_run_without_mps_fallback_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    pose = _quarter_turn_pose(device="mps")
    points = torch.ones((1, 4, 3), device="mps", requires_grad=True)
    output = batch_transform_points(pose.position, pose.quaternion, points)
    inverse = batch_transform_points_inverse(pose.position, pose.quaternion, output)
    torch.testing.assert_close(inverse, points, atol=2e-5, rtol=0)
    output.square().sum().backward()
    assert points.grad is not None and points.grad.device.type == "mps"
