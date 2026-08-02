"""Focused portable coverage for concrete geometry helper functions."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.data.data_voxel import (
    is_voxel_valid,
    sample_voxel_sdf,
    sample_voxel_sdf_with_grad,
    voxel_idx_to_flat,
    world_to_voxel_idx,
)
from curobo._src.geom.data.helper_pose import (
    get_forward_quat,
    load_inv_position,
    load_inv_quat,
)
from curobo._src.geom.types import (
    batch_tensor_cube,
    tensor_capsule,
    tensor_cube,
    tensor_sphere,
)
from curobo._src.types.device_cfg import DeviceCfg


def test_voxel_index_helpers_preserve_c_order_and_bounds() -> None:
    index = torch.tensor([[0, 0, 0], [1, 2, 3], [-1, 0, 0]])
    dims = torch.tensor([2, 3, 4])
    torch.testing.assert_close(voxel_idx_to_flat(index, dims), torch.tensor([0, 23, -12]))
    assert is_voxel_valid(index, dims).tolist() == [True, True, False]
    # Warp int conversion truncates, rather than floors, negatives near zero.
    torch.testing.assert_close(
        world_to_voxel_idx(torch.tensor([-1.9, -0.1, 0.9]), dims, 1.0),
        torch.tensor([0, 1, 2]),
    )


def test_voxel_sampling_and_gradient_are_differentiable() -> None:
    features = torch.arange(8.0, requires_grad=True)
    point = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    nearest = sample_voxel_sdf(features, 0, torch.tensor([1, 1, 1]), [2, 2, 2], 99.0)
    torch.testing.assert_close(nearest, torch.tensor([7.0, 1.0]))
    out_of_bounds = sample_voxel_sdf(features, 0, torch.tensor([2, 0, 0]), [2, 2, 2], 99.0)
    torch.testing.assert_close(out_of_bounds, torch.tensor([99.0, 0.0]))

    sampled = sample_voxel_sdf_with_grad(features, 0, point, [2, 2, 2], 1.0, 99.0)
    torch.testing.assert_close(sampled, torch.tensor([3.5, 4.0, 2.0, 1.0]))
    sampled[0].backward()
    torch.testing.assert_close(point.grad, torch.tensor([4.0, 2.0, 1.0]))
    assert features.grad is not None and torch.all(features.grad > 0)


def test_voxel_sampling_handles_boundary_and_batches() -> None:
    features = torch.arange(8.0)
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sampled = sample_voxel_sdf_with_grad(features, 0, points, [2, 2, 2], 1.0, 99.0)
    assert sampled.shape == (2, 4)
    torch.testing.assert_close(sampled[0], torch.tensor([3.5, 4.0, 2.0, 1.0]))
    torch.testing.assert_close(sampled[1], torch.tensor([99.0, 0.0, 0.0, 0.0]))


def test_pose_helpers_follow_warp_xyzw_convention() -> None:
    poses = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]])
    torch.testing.assert_close(load_inv_position(poses, 0), torch.tensor([1.0, 2.0, 3.0]))
    inverse = load_inv_quat(poses, 0)
    torch.testing.assert_close(inverse, torch.tensor([5.0, 6.0, 7.0, 4.0]))
    torch.testing.assert_close(get_forward_quat(inverse), torch.tensor([-5.0, -6.0, -7.0, 4.0]))


def test_geometry_tensor_helpers_preserve_values_and_gradients() -> None:
    cfg = DeviceCfg()
    point = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    sphere = tensor_sphere(point, 0.4, device_cfg=cfg)
    capsule = tensor_capsule(point, point + 1, 0.2, device_cfg=cfg)
    torch.testing.assert_close(sphere, torch.tensor([1.0, 2.0, 3.0, 0.4]))
    torch.testing.assert_close(capsule, torch.tensor([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 0.2]))
    (sphere.sum() + capsule.sum()).backward()
    torch.testing.assert_close(point.grad, torch.full((3,), 3.0))

    dims, inverse = tensor_cube([1, 0, 0, 1, 0, 0, 0], [2, 3, 4], cfg)
    torch.testing.assert_close(dims, torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(inverse, torch.tensor([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    batch_dims, batch_inverse = batch_tensor_cube(
        [[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]], [[1, 1, 1], [2, 2, 2]], cfg
    )
    assert batch_dims.shape == (2, 3) and batch_inverse.shape == (2, 7)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_voxel_helpers_run_without_cpu_fallback_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    features = torch.arange(8.0, device="mps")
    result = sample_voxel_sdf_with_grad(
        features, 0, torch.zeros(3, device="mps"), [2, 2, 2], 1.0, 99.0
    )
    assert result.device.type == "mps"
