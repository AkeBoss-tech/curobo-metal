"""Portable behaviour checks for the V2 robot depth-segmentation facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.robot_segmenter import (
    RobotSegmenter,
    _mask_spheres_image_cdist,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _segmenter(*, device: str = "cpu", buffer: float | None = None) -> RobotSegmenter:
    return RobotSegmenter.from_robot_file(
        "franka.yml",
        collision_sphere_buffer=buffer,
        distance_threshold=0.005,
        use_cuda_graph=True,
        device_cfg=DeviceCfg(torch.device(device)),
    )


def _center_camera(segmenter: RobotSegmenter, *, batch: int = 1) -> tuple[CameraObservation, JointState]:
    q = segmenter.kinematics.default_joint_position.repeat(batch, 1)
    state = segmenter.kinematics.compute_kinematics(
        JointState.from_position(q, segmenter.kinematics.joint_names)
    )
    sphere = state.robot_spheres[0, 0, 0]
    # The central ray points along camera +z.  Position the camera so that it
    # lands precisely at the first real Franka collision sphere centre.
    camera_position = sphere[:3] - torch.tensor(
        [0.0, 0.0, 0.30], device=q.device, dtype=q.dtype
    )
    depth = torch.zeros((1, 3, 3), dtype=q.dtype, device=q.device)
    depth[0, 1, 1] = 0.30
    depth[0, 0, 0] = 1.0
    camera = CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor(
            [[5.0, 0.0, 1.0], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=q.dtype,
            device=q.device,
        ),
        pose=Pose(
            camera_position.unsqueeze(0),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=q.dtype, device=q.device),
        ),
        depth_to_meter=1.0,
    )
    return camera, JointState.from_position(q, segmenter.kinematics.joint_names)


def test_robot_segmenter_projects_in_robot_frame_and_masks_only_valid_depth():
    segmenter = _segmenter()
    camera, state = _center_camera(segmenter)

    points = segmenter.get_pointcloud_from_depth(camera)
    assert points.shape == (1, 3, 3, 3)
    # Projection alone stays in the camera frame; the configured pose is
    # applied only for robot-sphere distance testing.
    torch.testing.assert_close(points[0, 1, 1], torch.tensor([0.0, 0.0, 0.30]))
    mask, filtered = segmenter.get_robot_mask(camera, state)
    assert mask.dtype == torch.bool and mask.shape == camera.depth_image.shape
    assert mask[0, 1, 1]
    assert not mask[0, 2, 2]  # invalid zero-depth points are never robot pixels
    assert filtered[0, 1, 1] == 0.0
    assert filtered[0, 0, 0] == camera.depth_image[0, 0, 0]
    assert segmenter.ready and camera.projection_rays is not None


def test_robot_segmenter_broadcasts_one_camera_across_active_joint_batches():
    segmenter = _segmenter()
    camera, state = _center_camera(segmenter, batch=2)
    mask, filtered = segmenter.get_robot_mask_from_active_js(camera, state)
    assert mask.shape == filtered.shape == (2, 3, 3)
    torch.testing.assert_close(mask[0], mask[1])
    torch.testing.assert_close(filtered[0], filtered[1])


def test_collision_sphere_buffer_is_applied_without_mutating_caller_state():
    baseline = _segmenter(buffer=None)
    buffered = _segmenter(buffer=0.01)
    base_radius = baseline.kinematics.robot_spheres[0, 3]
    buffered_radius = buffered.kinematics.robot_spheres[0, 3]
    torch.testing.assert_close(buffered_radius, base_radius + 0.01)


def test_low_level_sphere_mask_has_empty_and_invalid_depth_semantics():
    depth = torch.tensor([[[1.0, 0.0]]])
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    spheres = torch.empty((1, 0, 4))
    empty_mask, empty_filtered = _mask_spheres_image_cdist(depth, spheres, points, 0.1)
    assert not empty_mask.any()
    torch.testing.assert_close(empty_filtered, depth)

    spheres = torch.tensor([[[0.0, 0.0, 0.0, 0.5]]])
    mask, filtered = _mask_spheres_image_cdist(depth, spheres, points, 0.0)
    assert mask.tolist() == [[[True, False]]]
    assert filtered.tolist() == [[[0.0, 0.0]]]


def test_robot_segmenter_validates_public_depth_batch_contract():
    segmenter = _segmenter()
    camera, state = _center_camera(segmenter)
    camera.depth_image = camera.depth_image[0]
    with pytest.raises(ValueError, match="batch, height, width"):
        segmenter.get_robot_mask(camera, state[0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_robot_segmenter_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    segmenter = _segmenter(device="mps")
    camera, state = _center_camera(segmenter)
    mask, filtered = segmenter.get_robot_mask(camera, state)
    assert mask.device.type == filtered.device.type == "mps"
    assert mask.any()
