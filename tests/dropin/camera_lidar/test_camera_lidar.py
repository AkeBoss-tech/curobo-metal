"""Portable lifecycle coverage for pinned camera and LiDAR value models."""

from __future__ import annotations

import pytest
import torch

from curobo._src.types.camera import (
    CameraObservation,
    extract_depth_from_structured_pointcloud,
    get_projection_rays,
    project_depth_using_rays,
)
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose


def _pose(device: str = "cpu") -> Pose:
    return Pose(
        torch.zeros((1, 3), device=device),
        torch.tensor(((1.0, 0.0, 0.0, 0.0),), device=device),
    )


def _camera(device: str = "cpu", *, requires_grad: bool = False) -> CameraObservation:
    depth = torch.tensor(((1.0, 2.0), (3.0, 4.0)), device=device, requires_grad=requires_grad)
    intrinsics = torch.tensor(((2.0, 0.0, 0.5), (0.0, 2.0, 0.5), (0.0, 0.0, 1.0)), device=device)
    return CameraObservation(
        rgb_image=torch.zeros((2, 2, 3), dtype=torch.uint8, device=device),
        depth_image=depth,
        intrinsics=intrinsics,
        pose=_pose(device),
        resolution=[2, 2],
        depth_to_meter=0.5,
    )


def test_camera_projection_validation_lifecycle_and_serialization(tmp_path):
    camera = _camera(requires_grad=True).validate(
        require_depth=True, require_intrinsics=True, require_pose=True, require_rgb=True
    )
    assert camera.device.type == "cpu"
    rays = get_projection_rays(2, 2, camera.intrinsics, camera.depth_to_meter)
    cloud = project_depth_using_rays(camera.depth_image, rays)
    assert cloud.shape == (1, 2, 2, 3)
    cloud.sum().backward()
    assert camera.depth_image.grad is not None
    camera.update_projection_rays()
    assert torch.allclose(camera.get_pointcloud(), cloud.detach())
    assert torch.equal(
        extract_depth_from_structured_pointcloud(cloud.detach()),
        camera.depth_image.detach().unsqueeze(0) * camera.depth_to_meter,
    )

    copied = CameraObservation(name="target")
    assert copied.copy_(camera) is copied
    assert copied.name == "target"
    assert copied.depth_image.data_ptr() != camera.depth_image.data_ptr()
    detached = camera.detach()
    assert not detached.depth_image.requires_grad
    camera_path = tmp_path / "camera.pt"
    camera.save_to_file(camera_path)
    restored = CameraObservation.load_from_file(camera_path)
    assert restored.name == camera.name
    assert torch.equal(restored.depth_image, camera.depth_image.detach())
    assert torch.equal(restored.pose.position, camera.pose.position)


def test_camera_mutation_methods_keep_upstream_non_fluent_contracts():
    camera = _camera()
    assert camera.filter_depth(1.5) is None
    torch.testing.assert_close(camera.depth_image, torch.tensor(((0.0, 2.0), (3.0, 4.0))))
    assert camera.update_projection_rays() is None
    assert camera.projection_rays is not None


def test_camera_observation_rejects_incompatible_geometry_and_stack():
    camera = _camera()
    camera.projection_rays = torch.zeros((1, 3, 2, 3))
    with pytest.raises(ValueError, match="spatial"):
        camera.validate()
    with pytest.raises(ValueError, match="positive"):
        CameraObservation(depth_to_meter=0)
    with pytest.raises(ValueError, match="focal"):
        get_projection_rays(2, 2, torch.tensor(((0.0, 0, 0), (0, 1, 0), (0, 0, 1))))
    with pytest.raises(ValueError, match="different depth_image"):
        CameraObservation(depth_image=torch.ones(2, 2)).stack(CameraObservation())
    with pytest.raises(ValueError, match="pose is required"):
        CameraObservation(depth_image=torch.ones(2, 2), intrinsics=torch.eye(3)).get_pointcloud(project_to_pose=True)


def _lidar(device: str = "cpu", *, requires_grad: bool = False) -> LidarObservation:
    ranges = torch.tensor((((1.0, 2.0, 3.0, 4.0),),), device=device, requires_grad=requires_grad)
    return LidarObservation(
        range_image=ranges,
        rgb_image=torch.zeros((1, 1, 4, 3), dtype=torch.uint8, device=device),
        valid_range_m=torch.tensor(((1.5, 3.5),), device=device),
        elevation_range_rad=torch.zeros((1, 2), device=device),
        pose=_pose(device),
    )


def test_lidar_validation_pointcloud_lifecycle_and_serialization(tmp_path):
    lidar = _lidar(requires_grad=True).validate(
        require_range=True, require_rgb=True, require_pose=True, require_calibration=True
    )
    mask = lidar.valid_mask()
    assert mask.tolist() == [[[False, True, True, False]]]
    points = lidar.to_pointcloud()
    assert points.shape == (1, 1, 4, 3)
    assert torch.equal(points[0, 0, 0], torch.zeros(3))
    assert torch.allclose(points[0, 0, 1], torch.tensor((0.0, -2.0, 0.0)), atol=1e-6)
    points.sum().backward()
    assert lidar.range_image.grad is not None
    copied = LidarObservation(name="target").copy_(lidar)
    assert copied.range_image.data_ptr() != lidar.range_image.data_ptr()
    path = tmp_path / "lidar.pt"
    lidar.save_to_file(path)
    restored = LidarObservation.load_from_file(path)
    assert torch.equal(restored.range_image, lidar.range_image.detach())
    assert torch.equal(restored.valid_range_m, lidar.valid_range_m)


def test_lidar_rejects_nonportable_layouts_before_external_integration():
    with pytest.raises(ValueError, match="shape"):
        LidarObservation(range_image=torch.ones(2, 2)).validate(require_range=True)
    with pytest.raises(TypeError, match="uint8"):
        LidarObservation(
            range_image=torch.ones((1, 2, 2)), rgb_image=torch.ones((1, 2, 2, 3))
        ).validate()
    with pytest.raises(ValueError, match="planar"):
        LidarObservation(
            range_image=torch.ones((1, 1, 2)),
            valid_range_m=torch.tensor(((0.0, 2.0),)),
            elevation_range_rad=torch.tensor(((0.0, 0.1),)),
        ).validate(require_calibration=True)
    with pytest.raises(ValueError, match="pose is required"):
        _lidar().to_pointcloud(project_to_pose=True) if False else LidarObservation(
            range_image=torch.ones((1, 1, 2)), valid_range_m=torch.tensor(((0.0, 2.0),)),
            elevation_range_rad=torch.zeros((1, 2)),
        ).to_pointcloud(project_to_pose=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_camera_and_lidar_lifecycle_runs_without_cpu_fallback_on_mps():
    camera = _camera("mps", requires_grad=True)
    output = camera.get_pointcloud().square().sum()
    output.backward()
    assert camera.depth_image.grad.device.type == "mps"
    lidar = _lidar("mps", requires_grad=True)
    world_points = lidar.to_pointcloud(project_to_pose=True)
    world_points.sum().backward()
    assert lidar.range_image.grad.device.type == "mps"
