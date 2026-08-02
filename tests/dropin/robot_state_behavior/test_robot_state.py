"""Portable RobotState lifecycle coverage against the pinned state contract."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import ToolPose


def _fk_state(batch: int, horizon: int, *, offset: float = 0.0) -> KinematicsState:
    position = torch.arange(batch * horizon * 3, dtype=torch.float32).reshape(batch, horizon, 1, 3)
    position = position + offset
    quaternion = torch.zeros(batch, horizon, 1, 4)
    quaternion[..., 0] = 1.0
    return KinematicsState(
        tool_poses=ToolPose(["tool"], position, quaternion),
        tool_jacobians=torch.full((batch, horizon, 1, 6, 2), offset),
        robot_spheres=torch.full((batch, horizon, 2, 4), offset),
        robot_com=torch.full((batch, horizon, 4), offset),
    )


def _state(
    *, batch: int = 2, seeds: int = 3, merged_fk: bool = False, offset: float = 0.0
) -> RobotState:
    joints = JointState.from_position(
        torch.full((batch, seeds, 2), offset), ["j0", "j1"]
    )
    fk = _fk_state(batch * seeds if merged_fk else batch, 1 if merged_fk else seeds, offset=offset)
    return RobotState(joints, torch.full((batch, seeds, 2), offset), fk)


def test_batch_seed_copy_updates_unmerged_kinematics_and_all_derivatives() -> None:
    target, source = _state(offset=0.0), _state(offset=5.0)
    batch = torch.tensor([0, 1])
    seed = torch.tensor([1, 2])
    target.copy_at_batch_seed_indices(source, batch, seed)

    torch.testing.assert_close(target.joint_state.position[batch, seed], source.joint_state.position[batch, seed])
    torch.testing.assert_close(target.joint_torque[batch, seed], source.joint_torque[batch, seed])
    torch.testing.assert_close(target.robot_spheres[batch, seed], source.robot_spheres[batch, seed])
    torch.testing.assert_close(target.cuda_robot_model_state.tool_jacobians[batch, seed], source.cuda_robot_model_state.tool_jacobians[batch, seed])
    torch.testing.assert_close(target.cuda_robot_model_state.robot_com[batch, seed], source.cuda_robot_model_state.robot_com[batch, seed])
    torch.testing.assert_close(target.tool_poses.position[batch, seed], source.tool_poses.position[batch, seed])
    torch.testing.assert_close(target.joint_state.velocity[batch, seed], source.joint_state.velocity[batch, seed])


def test_batch_seed_copy_detects_flattened_fk_storage() -> None:
    target, source = _state(merged_fk=True, offset=0.0), _state(merged_fk=True, offset=7.0)
    batch = torch.tensor([0, 1])
    seed = torch.tensor([2, 0])
    linear = batch * 3 + seed
    target.copy_at_batch_seed_indices(source, batch, seed)

    torch.testing.assert_close(target.robot_spheres[linear], source.robot_spheres[linear])
    torch.testing.assert_close(target.tool_poses.quaternion[linear], source.tool_poses.quaternion[linear])
    torch.testing.assert_close(target.cuda_robot_model_state.tool_jacobians[linear], source.cuda_robot_model_state.tool_jacobians[linear])
    assert target.robot_spheres[1].eq(0).all()


def test_clone_detach_contiguous_and_to_preserve_value_semantics() -> None:
    state = _state(offset=3.0)
    state.joint_state.position.requires_grad_()
    state.joint_torque.requires_grad_()
    state.cuda_robot_model_state.robot_spheres.requires_grad_()
    cloned = state.clone()
    cloned.joint_state.position.zero_()
    assert state.joint_state.position.count_nonzero() > 0
    assert cloned.robot_spheres.data_ptr() != state.robot_spheres.data_ptr()

    detached = state.detach()
    assert state.joint_state.position.requires_grad
    assert not detached.joint_state.position.requires_grad
    assert not detached.joint_torque.requires_grad
    assert not detached.robot_spheres.requires_grad

    noncontiguous = RobotState(
        JointState.from_position(state.joint_state.position.transpose(0, 1)),
        state.joint_torque.transpose(0, 1),
        KinematicsState(
            ToolPose(["tool"], state.tool_poses.position.transpose(0, 1), state.tool_poses.quaternion.transpose(0, 1)),
            state.cuda_robot_model_state.tool_jacobians.transpose(0, 1),
            state.robot_spheres.transpose(0, 1),
            state.cuda_robot_model_state.robot_com.transpose(0, 1),
        ),
    )
    contiguous = noncontiguous.contiguous()
    assert contiguous.joint_state.position.is_contiguous()
    assert contiguous.joint_torque.is_contiguous()
    assert contiguous.robot_spheres.is_contiguous()

    converted = state.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert converted.dtype == torch.float64
    assert converted.robot_spheres.dtype == torch.float64
    assert state.dtype == torch.float32
    assert state.to() is state


def test_index_and_copy_report_incompatible_optional_buffers_precisely() -> None:
    target, source = _state(offset=0.0), _state(offset=2.0)
    selected = source[torch.tensor([1, 0])]
    assert selected.joint_state.position.shape == (2, 3, 2)
    assert selected.robot_spheres.shape == (2, 3, 2, 4)
    target.copy_only_index(source, torch.tensor([1]))
    torch.testing.assert_close(target.robot_spheres[1], source.robot_spheres[1])

    target.joint_torque = torch.zeros_like(target.joint_torque)
    source.joint_torque = None
    with pytest.raises(ValueError, match="missing joint_torque"):
        target.copy_(source)
    with pytest.raises(ValueError, match="same shape"):
        _state().copy_at_batch_seed_indices(_state(), torch.tensor([0]), torch.tensor([0, 1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_lifecycle_without_cpu_fallback() -> None:
    source = _state(offset=4.0).to(DeviceCfg(torch.device("mps"), torch.float32))
    target = _state().to(DeviceCfg(torch.device("mps"), torch.float32))
    batch = torch.tensor([1], device="mps")
    seed = torch.tensor([2], device="mps")
    target.copy_at_batch_seed_indices(source, batch, seed).requires_grad_()
    assert target.device.type == target.robot_spheres.device.type == "mps"
    assert target.joint_state.position.requires_grad
    torch.testing.assert_close(target.robot_spheres[1, 2].cpu(), source.robot_spheres[1, 2].cpu())
