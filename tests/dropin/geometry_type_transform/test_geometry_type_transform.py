"""Portable value-level regression coverage for geometry and transforms."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.transform import (
    BatchTransformPoint,
    MatrixToQuaternion,
    PoseInverse,
    TransformPoint,
    get_inv_transform,
    matrix_to_quaternion,
    pose_inverse,
    pose_multiply,
    pose_to_affine_matrix,
    pose_to_matrix,
    quaternion_rate_to_axis_angle_rate,
    quaternion_to_matrix,
    transform_point_inverse,
    transform_points,
)
from curobo._src.geom.types import Mesh, SceneCfg, Sphere, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg


def test_matrix_and_pose_helpers_cover_unbatched_batched_and_output_buffers() -> None:
    position = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    matrix_buffer = torch.empty(2, 4, 4)
    matrix = pose_to_matrix(position, quaternion, matrix_buffer)
    assert matrix.data_ptr() == matrix_buffer.data_ptr()
    torch.testing.assert_close(matrix[..., :3, 3], position)
    affine = pose_to_affine_matrix(position, quaternion)
    assert affine.shape == (2, 3, 4)
    torch.testing.assert_close(matrix_to_quaternion(quaternion_to_matrix(quaternion)), quaternion)

    inverse_position, inverse_quaternion = pose_inverse(position, quaternion)
    identity_position, identity_quaternion = pose_multiply(position, quaternion, inverse_position, inverse_quaternion)
    torch.testing.assert_close(identity_position, torch.zeros_like(position), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(identity_quaternion, torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]))


def test_matrix_inverse_and_point_inverse_use_the_pinned_matrix_signature() -> None:
    rotation = quaternion_to_matrix(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    translation = torch.tensor([[2.0, -1.0, 0.0]])
    inverse_rotation, inverse_translation = get_inv_transform(rotation, translation)
    torch.testing.assert_close(inverse_rotation @ rotation, torch.eye(3).reshape(1, 3, 3))
    point = torch.tensor([[2.0, 1.0, 0.0]])
    expected = torch.matmul(point, inverse_rotation.transpose(-1, -2)) + inverse_translation
    torch.testing.assert_close(transform_point_inverse(point, rotation, translation), expected)


def test_function_facades_retain_first_order_autograd() -> None:
    position = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    quaternion = torch.tensor([[1.0, 0.1, 0.0, 0.0]], requires_grad=True)
    points = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    transformed = TransformPoint.apply(position, quaternion, points)
    transformed.sum().backward()
    assert position.grad is not None and quaternion.grad is not None and points.grad is not None

    output_buffer = torch.empty_like(transformed)
    buffered = TransformPoint.apply(position.detach(), quaternion.detach(), points.detach(), output_buffer)
    assert buffered.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(buffered, transformed.detach())

    position.grad = quaternion.grad = points.grad = None
    batched = BatchTransformPoint.apply(position, quaternion, points.unsqueeze(0))
    batched.sum().backward()
    assert position.grad is not None and quaternion.grad is not None and points.grad is not None

    rotation = quaternion_to_matrix(quaternion.detach()).requires_grad_()
    value = MatrixToQuaternion.apply(rotation)
    value.sum().backward()
    assert rotation.grad is not None

    inverse_position, inverse_quaternion = PoseInverse.apply(position.detach(), quaternion.detach())
    recomposed_position, recomposed_quaternion = pose_multiply(
        position.detach(), quaternion.detach(), inverse_position, inverse_quaternion
    )
    torch.testing.assert_close(recomposed_position, torch.zeros_like(position))
    torch.testing.assert_close(recomposed_quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_quaternion_rate_uses_pinned_residual_convention() -> None:
    rate = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(quaternion_rate_to_axis_angle_rate(rate, identity), torch.tensor([[2.0, 3.0, 4.0]]))


def test_scene_clone_is_independent_and_preserves_tensor_graph() -> None:
    feature = torch.arange(8.0, requires_grad=True)
    world = SceneCfg(
        sphere=[Sphere("ball", position=[1, 0, 0], radius=0.2)],
        voxel=[VoxelGrid("vox", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2], voxel_size=1.0, feature_tensor=feature)],
    )
    clone = world.clone()
    clone.sphere[0].pose[0] = 5.0
    clone.voxel[0].feature_tensor[0] = -4.0
    assert world.sphere[0].pose[0] == 1
    assert world.voxel[0].feature_tensor[0].item() == 0
    clone.voxel[0].feature_tensor.sum().backward()
    assert feature.grad is not None


def test_empty_point_cloud_mesh_is_well_formed() -> None:
    mesh = Mesh.from_pointcloud([], pose=None)
    assert torch.as_tensor(mesh.vertices).shape == (1, 3)
    assert torch.as_tensor(mesh.faces).shape == (1, 3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_transforms_and_scene_values_stay_on_mps_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = torch.device("mps")
    position = torch.zeros(1, 3, device=device, requires_grad=True)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    points = torch.ones(1, 2, 3, device=device, requires_grad=True)
    out = transform_points(position, quaternion, points)
    assert out.device.type == "mps"
    out.sum().backward()
    assert position.grad is not None and points.grad is not None
    grid = VoxelGrid("vox", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2], voxel_size=1.0, feature_tensor=torch.ones(8, device=device), device_cfg=DeviceCfg(device="mps"))
    assert grid.clone().feature_tensor.device.type == "mps"
