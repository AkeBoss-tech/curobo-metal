"""High-value CPU/MPS lifecycle coverage for the portable V2 ``Pose`` type."""

from __future__ import annotations

import pytest
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import (
    Pose,
    batch_transform_points,
    batch_transform_points_inverse,
    transform_points,
)


def _z_pose(*, device: str = "cpu") -> Pose:
    half = torch.tensor(2.0**-0.5, device=device)
    return Pose(
        torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]], device=device).requires_grad_(),
        torch.stack((
            torch.stack((half, torch.zeros_like(half), torch.zeros_like(half), half)),
            torch.tensor((1.0, 0.0, 0.0, 0.0), device=device),
        )).requires_grad_(),
        name="camera",
    )


def test_pairwise_and_cloud_transforms_are_differentiable_and_do_not_cross_product() -> None:
    pose = _z_pose()
    pairwise = torch.tensor([[2.0, 0.0, 1.0], [0.0, 1.0, -1.0]], requires_grad=True)
    transformed = pose.transform_points(pairwise)
    assert transformed.shape == (2, 3)
    torch.testing.assert_close(
        transformed,
        torch.tensor([[1.0, 4.0, 0.0], [-2.0, 1.5, 2.0]]),
        atol=1e-6,
        rtol=0,
    )

    raw = transform_points(pose.position, pose.quaternion, pairwise)
    assert raw.shape == (2, 3)
    torch.testing.assert_close(raw, transformed)

    cloud = torch.arange(24.0).reshape(2, 4, 3).requires_grad_()
    cloud_out = torch.empty_like(cloud)
    transformed_cloud = pose.batch_transform_points(cloud, cloud_out)
    assert transformed_cloud is cloud_out
    restored = batch_transform_points_inverse(
        pose.position, pose.quaternion, transformed_cloud
    )
    torch.testing.assert_close(restored, cloud, atol=3e-6, rtol=0)
    raw_cloud = batch_transform_points(pose.position, pose.quaternion, cloud)
    torch.testing.assert_close(raw_cloud, transformed_cloud)

    (transformed.square().sum() + transformed_cloud.square().sum()).backward()
    assert pairwise.grad is not None and torch.isfinite(pairwise.grad).all()
    assert cloud.grad is not None and torch.isfinite(cloud.grad).all()
    assert pose.position.grad is not None and torch.isfinite(pose.position.grad).all()
    assert pose.quaternion.grad is not None and torch.isfinite(pose.quaternion.grad).all()


def test_rotation_cache_is_coherent_after_copy_and_index_assignment() -> None:
    base = _z_pose()
    matrix_backed = Pose.from_matrix(base.get_matrix())
    matrix_backed.name = "matrix_camera"
    identity = Pose(
        torch.zeros_like(base.position),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        name="identity",
    )

    matrix_backed.copy_(identity)
    torch.testing.assert_close(matrix_backed.get_rotation(), torch.eye(3).expand(2, -1, -1))
    matrix_backed[0] = base[0]
    torch.testing.assert_close(matrix_backed.get_rotation()[0], base.get_rotation()[0])
    torch.testing.assert_close(matrix_backed.get_rotation()[1], torch.eye(3))

    indexed = matrix_backed.get_index(0)
    assert indexed.name == "matrix_camera"
    assert indexed.position.shape == (1, 3)
    assert indexed.rotation is not None and indexed.rotation.shape == (1, 3, 3)


def test_clone_detach_and_to_preserve_pose_metadata_and_value_independence() -> None:
    base = _z_pose()
    matrix_backed = Pose.from_matrix(base.get_matrix())
    matrix_backed.name = "tool"
    clone = matrix_backed.clone()
    detached = matrix_backed.detach()
    assert clone.name == detached.name == "tool"
    assert clone.position.data_ptr() != matrix_backed.position.data_ptr()
    assert clone.quaternion.data_ptr() != matrix_backed.quaternion.data_ptr()
    assert detached.position.requires_grad is False
    assert detached.quaternion.requires_grad is False
    assert detached.rotation is not None and detached.rotation.requires_grad is False

    converted = clone.to(DeviceCfg(device="cpu", dtype=torch.float64))
    assert converted is clone
    assert clone.position.dtype is torch.float64
    assert clone.quaternion.dtype is torch.float64
    assert clone.rotation is not None and clone.rotation.dtype is torch.float64


def test_transform_shape_errors_are_explicit() -> None:
    pose = _z_pose()
    with pytest.raises(ValueError, match="one pose or one pose per flattened point"):
        pose.transform_points(torch.zeros(3, 3))
    with pytest.raises(ValueError, match="flattened pose batch"):
        pose.batch_transform_points(torch.zeros(3, 2, 3))


def test_pose_lifecycle_runs_without_mps_fallback_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    pose = _z_pose(device="mps")
    cloud = torch.ones((2, 3, 3), device="mps", requires_grad=True)
    output = pose.batch_transform_points(cloud)
    restored = pose.batch_transform_points_inverse(output)
    torch.testing.assert_close(restored, cloud, atol=2e-5, rtol=0)
    output.square().sum().backward()
    assert cloud.grad is not None and cloud.grad.device.type == "mps"
    assert pose.position.grad is not None and pose.position.grad.device.type == "mps"
