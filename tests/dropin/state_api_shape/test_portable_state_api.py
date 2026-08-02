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
    right = Pose.from_list([0, 2, 0, 1, 0, 0, 0])
    combined = pose_multiply(left, right)
    torch.testing.assert_close(combined.position, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(pose_inverse(left).multiply(left).position, torch.zeros(1, 3))
    assert pose_to_matrix(left).shape == (1, 4, 4)
    assert pose_to_affine_matrix(left).shape == (1, 3, 4)

    rotation = quaternion_to_matrix(left.quaternion)
    recovered = matrix_to_quaternion(rotation)
    torch.testing.assert_close(recovered.abs(), left.quaternion)
    torch.testing.assert_close(angular_distance_phi3(left.quaternion, recovered), torch.zeros(1))
    torch.testing.assert_close(angular_distance_axis_angle(left.quaternion, recovered), torch.zeros(1))

    local = torch.tensor([[[1.0, 0.0, 0.0]]])
    world = left.batch_transform_points(local)
    torch.testing.assert_close(batch_transform_points_inverse(left, world), local)
    combined.position.sum().backward()
    assert position.grad is not None
