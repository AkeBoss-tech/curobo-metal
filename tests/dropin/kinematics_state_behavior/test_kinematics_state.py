"""Lifecycle coverage for the portable FK output state record."""

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_state import KinematicsState, ToolPose
from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo.types import DeviceCfg


def _state(*, requires_grad: bool = False) -> KinematicsState:
    position = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 3)
    quaternion = torch.zeros(2, 3, 2, 4)
    quaternion[..., 0] = 1.0
    jacobian = torch.arange(2 * 3 * 2 * 6 * 4, dtype=torch.float32).reshape(2, 3, 2, 6, 4)
    spheres = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)
    com = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    if requires_grad:
        position.requires_grad_()
        quaternion.requires_grad_()
        jacobian.requires_grad_()
        spheres.requires_grad_()
        com.requires_grad_()
    return KinematicsState(
        ToolPose(["left", "right"], position, quaternion),
        jacobian,
        spheres,
        com,
        RobotCollisionGeometry(torch.tensor([0, 1, 1, 0], dtype=torch.int64), 2),
    )


def test_state_uses_canonical_tool_pose_layout_and_link_helpers() -> None:
    state = _state()
    assert state.tool_frames == ["left", "right"]
    assert state.batch_size == len(state) == 2
    assert state.horizon == 3
    assert state.device.type == "cpu"
    assert state.dtype == torch.float32
    assert state.get_link_spheres() is state.robot_spheres

    pose = state.tool_poses.get_link_pose("right")
    assert pose.position.shape == (6, 3)
    assert state.tool_poses.to_dict()["left"].position.shape == (6, 3)
    assert state.tool_poses.as_goal().position.shape == (2, 3, 2, 1, 3)


def test_clone_detach_copy_and_contiguous_keep_expected_ownership() -> None:
    state = _state(requires_grad=True)
    clone = state.clone()
    assert clone.robot_spheres.data_ptr() != state.robot_spheres.data_ptr()
    assert (
        clone.robot_collision_geometry.link_sphere_idx_map.data_ptr()
        != state.robot_collision_geometry.link_sphere_idx_map.data_ptr()
    )
    clone.robot_spheres.zero_()
    assert state.robot_spheres.count_nonzero() > 0

    detached = state.detach()
    assert not detached.tool_poses.position.requires_grad
    assert not detached.robot_spheres.requires_grad
    assert not detached.robot_collision_geometry.link_sphere_idx_map.requires_grad

    target = state.clone()
    target.robot_spheres.zero_()
    target.copy_(state)
    torch.testing.assert_close(target.robot_spheres, state.robot_spheres)
    with pytest.raises(ValueError, match="missing source field"):
        target.copy_(KinematicsState(robot_spheres=state.robot_spheres))

    noncontiguous = KinematicsState(
        ToolPose(
            state.tool_frames,
            state.tool_poses.position.transpose(0, 1),
            state.tool_poses.quaternion.transpose(0, 1),
        ),
        state.tool_jacobians.transpose(0, 1),
        state.robot_spheres.transpose(0, 1),
        state.robot_com.transpose(0, 1),
    )
    assert not noncontiguous.robot_spheres.is_contiguous()
    assert noncontiguous.contiguous().robot_spheres.is_contiguous()


def test_index_preserves_pinned_tool_pose_and_raw_tensor_view_rules() -> None:
    state = _state()
    selected = state[1]
    # ToolPose follows its public singleton-batch convention; raw fields use
    # ordinary PyTorch integer indexing exactly like pinned cuRobo.
    assert selected.tool_poses.position.shape == (1, 3, 2, 3)
    assert selected.tool_jacobians.shape == (3, 2, 6, 4)
    assert selected.robot_spheres.shape == (3, 5, 4)
    with torch.no_grad():
        selected.robot_spheres[0, 0, 0] = -99
    assert state.robot_spheres[1, 0, 0, 0].item() == -99

    batch = state[torch.tensor([1, 0])]
    assert batch.tool_poses.position.shape == (2, 3, 2, 3)
    assert batch.robot_com.shape == (2, 3, 4)


def test_to_returns_independent_portable_tensor_and_geometry_payloads() -> None:
    state = _state()
    converted = state.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert converted is not state
    assert converted.dtype == torch.float64
    assert converted.tool_poses.position.dtype == torch.float64
    assert converted.robot_collision_geometry.link_sphere_idx_map.dtype == torch.int64
    assert converted.robot_collision_geometry.link_sphere_idx_map.device.type == "cpu"
    assert state.dtype == torch.float32
    assert state.to() is state


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_to_mps_without_cpu_fallback() -> None:
    state = _state()
    converted = state.to(DeviceCfg(torch.device("mps"), torch.float32))
    assert converted.device.type == "mps"
    assert converted.tool_jacobians.device.type == "mps"
    assert converted.robot_collision_geometry.link_sphere_idx_map.device.type == "mps"
    torch.testing.assert_close(converted.robot_com.cpu(), state.robot_com)
