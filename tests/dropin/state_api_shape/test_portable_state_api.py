"""Portable state/type surface checks that do not require CUDA runtime ABI."""

from __future__ import annotations

import pytest
import torch

from curobo._src.state import JointState
from curobo._src.state.state_base import State
from curobo._src.state.state_joint import (
    append_joints_to_state,
    joint_state_to_tensor,
    trim_joint_state_trajectory,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import (
    Pose,
    angular_distance_axis_angle,
    angular_distance_phi3,
    batch_transform_points_inverse,
    matrix_to_quaternion,
    pose_inverse,
    pose_multiply,
    pose_to_affine_matrix,
    pose_to_matrix,
    quaternion_to_matrix,
)
from curobo._src.types.camera import (
    extract_depth_from_structured_pointcloud,
    get_projection_rays,
    project_depth_using_rays,
)
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def test_state_module_reexports_preserve_tensor_graph_and_sequence_contract() -> None:
    position = torch.arange(12.0).reshape(1, 4, 3).requires_grad_()
    state = JointState.from_position(position, ["a", "b", "c"])
    assert isinstance(state, State)
    assert joint_state_to_tensor(state).shape == (1, 4, 12)
    assert trim_joint_state_trajectory(state, 1).position.shape == (1, 3, 3)

    extra = JointState.from_position(torch.ones(1, 4, 1), ["tool"])
    combined = append_joints_to_state(state, extra)
    combined.position.sum().backward()
    assert position.grad is not None
    assert combined.joint_names == ["a", "b", "c", "tool"]


def test_device_cfg_portable_integer_bool_clone_and_mps_boundary() -> None:
    cfg = DeviceCfg("cpu", torch.float64)
    assert cfg.to_int8_device([1]).dtype == torch.int8
    assert cfg.to_int32_device([1]).dtype == torch.int32
    assert cfg.to_int64_device([1]).dtype == torch.int64
    assert cfg.to_bool_device([1]).dtype == torch.bool
    assert cfg.clone() == cfg
    assert DeviceCfg.from_basic("mps", 0).device == torch.device("mps")

    with pytest.raises(TypeError, match="only float32"):
        DeviceCfg("mps", torch.float64)


def test_pose_function_reexports_are_differentiable_and_wxyz_consistent() -> None:
    position = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    left = Pose(position, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    right = Pose.from_list([0, 2, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu"))
    combined = pose_multiply(left, right)
    torch.testing.assert_close(combined.position, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(pose_inverse(left).multiply(left).position, torch.zeros(1, 3))
    assert pose_to_matrix(left).shape == (1, 4, 4)
    assert pose_to_affine_matrix(left).shape == (1, 3, 4)

    rotation = quaternion_to_matrix(left.quaternion)
    recovered = matrix_to_quaternion(rotation)
    torch.testing.assert_close(recovered.abs(), left.quaternion)
    torch.testing.assert_close(angular_distance_phi3(left.quaternion, recovered), torch.zeros(1))
    torch.testing.assert_close(
        angular_distance_axis_angle(left.quaternion, recovered), torch.zeros(1, 1)
    )

    local = torch.tensor([[[1.0, 0.0, 0.0]]])
    world = left.batch_transform_points(local)
    torch.testing.assert_close(batch_transform_points_inverse(left, world), local)
    combined.position.sum().backward()
    assert position.grad is not None


def test_explicit_inherited_state_and_pose_surface_preserves_metadata_and_autograd() -> None:
    position = torch.arange(6.0).reshape(2, 3).requires_grad_()
    state = JointState.from_position(position, ["a", "b", "c"])
    assert state.device == position.device
    assert state.dtype == position.dtype
    assert state.ndim == 2
    assert state.reorder(["c", "a"]).joint_names == ["c", "a"]
    state.view(1, 2, 3).position.sum().backward()
    assert position.grad is not None

    matrix = torch.eye(4)
    matrix[0, 0] = matrix[1, 1] = -1
    pose = Pose.from_matrix(matrix)
    assert pose.batch == 1
    assert pose.get_pose_vector().shape == (1, 7)
    torch.testing.assert_close(pose.inverse().position, torch.zeros(1, 3))
    torch.testing.assert_close(pose.get_rotation(), matrix[:3, :3].unsqueeze(0))
    assert Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu")).to_list() == [0.0] * 3 + [1.0, 0.0, 0.0, 0.0]


def test_tool_goal_sequence_metadata_and_camera_function_reexports() -> None:
    position = torch.zeros(2, 3, 1, 3)
    quaternion = torch.zeros(2, 3, 1, 4)
    quaternion[..., 0] = 1
    tool = ToolPose(["tool"], position, quaternion)
    assert (tool.batch_size, tool.horizon, tool.num_links, tool.ndim) == (2, 3, 1, 4)
    goal = tool.as_goal()
    assert isinstance(goal, GoalToolPose)
    assert (goal.batch_size, goal.horizon, goal.num_links, goal.num_goalset) == (2, 3, 1, 1)
    sequence = SequenceGoalToolPose(
        ["tool"], goal.position[:, 0].unsqueeze(0), goal.quaternion[:, 0].unsqueeze(0)
    )
    assert (sequence.num_frames, sequence.num_envs, sequence.num_links, sequence.num_goalset) == (1, 2, 1, 1)
    assert sequence.get_frame(0).shape == (2, 1, 1, 1, 3)

    intrinsics = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    rays = get_projection_rays(2, 2, intrinsics, depth_to_meter=1.0)
    cloud = project_depth_using_rays(torch.ones(1, 2, 2), rays)
    torch.testing.assert_close(extract_depth_from_structured_pointcloud(cloud), torch.ones(1, 2, 2))
