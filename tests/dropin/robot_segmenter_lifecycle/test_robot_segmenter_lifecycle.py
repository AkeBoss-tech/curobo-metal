"""Lifecycle checks for the portable robot-depth segmentation facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.robot_segmenter import RobotSegmenter
from curobo._src.state.state_joint import JointState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _segmenter(device: str = "cpu") -> RobotSegmenter:
    return RobotSegmenter.from_robot_file(
        "franka.yml",
        distance_threshold=0.005,
        use_cuda_graph=True,
        device_cfg=DeviceCfg(torch.device(device)),
    )


def _camera_and_state(
    segmenter: RobotSegmenter, *, x_offset: float = 0.0, dtype: torch.dtype = torch.float32
) -> tuple[CameraObservation, JointState]:
    q = segmenter.kinematics.default_joint_position.unsqueeze(0)
    state = segmenter.kinematics.compute_kinematics(
        JointState.from_position(q, segmenter.kinematics.joint_names)
    )
    sphere = state.robot_spheres[0, 0, 0]
    position = sphere[:3] - torch.tensor([0.0, 0.0, 0.30], device=q.device)
    position = position + torch.tensor([x_offset, 0.0, 0.0], device=q.device)
    depth = torch.zeros((1, 3, 3), device=q.device, dtype=dtype)
    depth[0, 1, 1] = 0.30
    camera = CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor(
            [[5.0, 0.0, 1.0], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]],
            device=q.device,
            dtype=dtype,
        ),
        pose=Pose(
            position.unsqueeze(0),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=q.device, dtype=q.dtype),
        ),
        depth_to_meter=1.0,
    )
    return camera, JointState.from_position(q, segmenter.kinematics.joint_names)


def test_segmenter_does_not_retain_the_first_mutable_camera_observation():
    segmenter = _segmenter()
    near_camera, state = _camera_and_state(segmenter)
    far_camera, _ = _camera_and_state(segmenter, x_offset=2.0)

    near_mask, _ = segmenter.get_robot_mask(near_camera, state)
    far_mask, far_depth = segmenter.get_robot_mask(far_camera, state)

    assert near_mask[0, 1, 1]
    assert not far_mask.any()
    torch.testing.assert_close(far_depth, far_camera.depth_image)


def test_projection_cache_refreshes_on_inplace_intrinsics_calibration_update():
    segmenter = _segmenter()
    camera, _ = _camera_and_state(segmenter)
    camera.depth_image[0, 1, 2] = 1.0

    initial = segmenter.get_pointcloud_from_depth(camera)
    initial_rays = segmenter._projection_rays
    assert initial_rays is not None
    camera.intrinsics[0, 0] = 10.0
    updated = segmenter.get_pointcloud_from_depth(camera)

    assert segmenter._projection_rays is not initial_rays
    assert not torch.allclose(initial[0, 1, 2], updated[0, 1, 2])
    assert segmenter.ready


def test_segmenter_reset_discards_camera_calibration_but_not_robot_model():
    segmenter = _segmenter()
    camera, state = _camera_and_state(segmenter)
    expected_dof = segmenter.kinematics.dof
    segmenter.get_robot_mask(camera, state)
    assert segmenter.ready

    segmenter.reset()

    assert not segmenter.ready
    assert segmenter._projection_rays is None
    assert segmenter.kinematics.dof == expected_dof
    mask, _ = segmenter.get_robot_mask(camera, state)
    assert mask[0, 1, 1] and segmenter.ready


def test_camera_depth_calibration_dtype_can_differ_from_robot_state_dtype():
    segmenter = _segmenter()
    camera, state = _camera_and_state(segmenter, dtype=torch.float64)
    mask, filtered = segmenter.get_robot_mask(camera, state)
    assert mask.dtype is torch.bool
    assert filtered.dtype is torch.float64
    assert mask[0, 1, 1]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_segmenter_mps_reuses_fresh_camera_frames_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    segmenter = _segmenter("mps")
    near_camera, state = _camera_and_state(segmenter)
    far_camera, _ = _camera_and_state(segmenter, x_offset=2.0)
    near_mask, _ = segmenter.get_robot_mask(near_camera, state)
    far_mask, _ = segmenter.get_robot_mask(far_camera, state)
    assert near_mask.device.type == far_mask.device.type == "mps"
    assert near_mask.any() and not far_mask.any()
