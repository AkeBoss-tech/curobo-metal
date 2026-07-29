from __future__ import annotations

import numpy as np
import pytest
import torch

from curobo_metal.types import (
    DeviceCfg,
    JointState,
    MotionGenResult,
    MotionGenStatus,
    Pose,
)


def test_joint_state_methods_preserve_derivatives_and_names() -> None:
    position = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    state = JointState.from_position(position, ["a", "b"])
    assert state.shape == (2, 2)
    assert state.dtype == torch.float64
    assert state.reorder(["b", "a"]).position.tolist() == [[2.0, 1.0], [4.0, 3.0]]
    sliced = state[0].unsqueeze(0).to(dtype=torch.float32)
    assert sliced.position.shape == sliced.velocity.shape == (1, 2)
    assert sliced.dtype == torch.float32
    assert sliced.joint_names == ["a", "b"]
    assert state.clone().get_state_tensor().shape == (2, 8)


def test_joint_state_numpy_and_device_cfg() -> None:
    cfg = DeviceCfg("cpu", torch.float64)
    state = JointState.from_numpy(["j"], np.array([[0.5]]), device_cfg=cfg)
    assert state.device == torch.device("cpu")
    assert cfg.as_torch_dict() == {"device": torch.device("cpu"), "dtype": torch.float64}
    assert cfg.is_same_torch_device(torch.device("cpu"))


def test_pose_round_trip_wxyz_and_batch_shape() -> None:
    pose = Pose.from_list([1, 2, 3, 1, 0, 0, 0], DeviceCfg("cpu", torch.float64))
    assert pose.position.shape == (1, 3)
    assert pose.quaternion.shape == (1, 4)
    recovered = Pose.from_matrix(pose.get_matrix())
    torch.testing.assert_close(recovered.get_matrix(), pose.get_matrix())
    assert recovered.tolist() == pytest.approx([1, 2, 3, 1, 0, 0, 0])
    assert pose.to(dtype=torch.float32).dtype == torch.float32


def test_pose_algebra_and_joint_state_trajectory_helpers() -> None:
    cfg = DeviceCfg("cpu", torch.float64)
    first = Pose.from_list([1, 0, 0, 1, 0, 0, 0], cfg)
    second = Pose.from_list([0, 2, 0, 1, 0, 0, 0], cfg)
    composed = first * second
    torch.testing.assert_close(
        composed.position, torch.tensor([[1., 2, 0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        (composed * composed.inverse()).get_matrix(),
        torch.eye(4, dtype=torch.float64).unsqueeze(0),
    )
    point = torch.tensor([[1., 1, 1]], dtype=torch.float64)
    torch.testing.assert_close(first.transform_points(point), point + first.position)

    position = torch.tensor([[0., 0], [1., 2], [2., 4]], dtype=torch.float64)
    differentiated = JointState(position, joint_names=["a", "b"]).finite_difference(.5)
    torch.testing.assert_close(
        differentiated.velocity,
        torch.tensor([[2., 4], [2., 4], [2., 4.]], dtype=torch.float64),
    )
    integrated = differentiated[0].integrate(.5)
    torch.testing.assert_close(
        integrated.position, torch.tensor([1., 2.], dtype=torch.float64)
    )


def test_device_cfg_integer_bool_helpers_and_clone() -> None:
    cfg = DeviceCfg("cpu", torch.float64)
    assert cfg.to_int32_device([1]).dtype == torch.int32
    assert cfg.to_int64_device([1]).dtype == torch.int64
    assert cfg.to_bool_device([1]).dtype == torch.bool
    assert cfg.clone() == cfg


def test_result_adapter_fields() -> None:
    state = JointState.from_position(torch.zeros(1, 2))
    result = MotionGenResult(True, MotionGenStatus.SUCCESS, state, state, 0.01)
    assert result.get_interpolated_plan() is state


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_types_move_to_mps() -> None:
    cfg = DeviceCfg("mps", torch.float32)
    state = JointState.from_position(torch.zeros(2)).to(cfg)
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(cfg)
    assert state.device.type == pose.device.type == "mps"
    with pytest.raises(TypeError, match="only float32"):
        DeviceCfg("mps", torch.float64)
