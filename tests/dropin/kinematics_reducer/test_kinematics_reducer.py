from __future__ import annotations

import pytest
import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_reducer import KinematicsReducer
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo_metal.config.robot import (
    CSpaceConfig,
    CollisionSphere,
    JointConfig,
    JointLimits,
    LinkConfig,
    RobotCfg,
)


def _branched_params(device_cfg: DeviceCfg = DeviceCfg()) -> KinematicsParams:
    robot = RobotCfg(
        name="branched",
        base_link="base",
        tool_frames=["tool", "collision_tip"],
        links=[LinkConfig("base"), LinkConfig("arm"), LinkConfig("tool"), LinkConfig("collision_tip")],
        joints=[
            JointConfig("arm_joint", "revolute", "base", "arm", limits=JointLimits(-2, 2, 2, 5)),
            JointConfig("tool_joint", "prismatic", "arm", "tool", limits=JointLimits(-1, 1, 2, 5)),
            JointConfig("collision_joint", "revolute", "base", "collision_tip", limits=JointLimits(-3, 3, 2, 5)),
        ],
        collision_spheres=[
            CollisionSphere("tool", (0, 0, 0), 0.1),
            CollisionSphere("collision_tip", (0, 0, 0), 0.2),
        ],
        cspace=CSpaceConfig(
            joint_names=["arm_joint", "tool_joint", "collision_joint"],
            default_joint_position=[0.1, 0.2, 0.3],
            max_acceleration=[4.0, 5.0, 6.0],
            max_jerk=[40.0, 50.0, 60.0],
            cspace_distance_weight=[1.0, 2.0, 3.0],
            null_space_weight=[3.0, 2.0, 1.0],
        ),
        collision_link_names=["tool", "collision_tip"],
        self_collision_ignore={"tool": ["collision_tip"], "collision_tip": ["tool"]},
        self_collision_buffer={"tool": 0.01, "collision_tip": 0.02},
        device_cfg=device_cfg,
    )
    return KinematicsParams(robot)


def test_reduce_prunes_collision_only_branch_and_records_locks():
    original = _branched_params()
    reduced = KinematicsReducer.reduce_dof(original, ["tool"])

    assert reduced.tool_frames == ["tool"]
    assert reduced.joint_names == ["arm_joint", "tool_joint"]
    assert reduced.all_link_names == ["base", "arm", "tool"]
    assert reduced.total_spheres == 1
    assert reduced.cspace.joint_names == ["arm_joint", "tool_joint"]
    torch.testing.assert_close(reduced.cspace.default_joint_position, torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(reduced.cspace.max_acceleration, torch.tensor([4.0, 5.0]))
    assert reduced.joint_limits.joint_names == ["arm_joint", "tool_joint"]
    assert reduced.lock_jointstate.joint_names == ["collision_joint"]
    torch.testing.assert_close(reduced.lock_jointstate.position, torch.tensor([0.3]))
    assert reduced.robot_cfg.collision_link_names == ["tool"]
    assert reduced.robot_cfg.self_collision_ignore == {"tool": []}

    model = Kinematics(KinematicsCfg(DeviceCfg(), ["tool"], reduced))
    state = model.compute_kinematics(
        JointState.from_position(torch.zeros(2, 2), joint_names=reduced.joint_names)
    )
    assert state.tool_poses.position.shape == (2, 1, 1, 3)


def test_reconstruct_preserves_batched_channels_metadata_and_order():
    original = _branched_params()
    reduced = KinematicsReducer.reduce_dof(original, ["tool"])
    position = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    state = JointState.from_position(position, joint_names=reduced.joint_names)
    state.velocity = torch.full_like(position, 7.0)
    state.dt = torch.tensor([0.1, 0.2])
    full = KinematicsReducer.reconstruct_joint_state(
        state, reduced.lock_jointstate, original.joint_names
    )
    assert full.joint_names == original.joint_names
    torch.testing.assert_close(full.position, torch.tensor([[1.0, 2.0, 0.3], [3.0, 4.0, 0.3]]))
    torch.testing.assert_close(full.velocity, torch.tensor([[7.0, 7.0, 0.0], [7.0, 7.0, 0.0]]))
    torch.testing.assert_close(full.dt, state.dt)
    gradient = torch.autograd.grad(full.position[..., :2].sum(), position)[0]
    torch.testing.assert_close(gradient, torch.ones_like(position))


def test_reconstruct_rejects_ambiguous_duplicate_lock_names():
    state = JointState.from_position(torch.zeros(1, 1), joint_names=["joint"])
    lock = JointState.from_position(torch.ones(1, 1), joint_names=["joint"])
    with pytest.raises(ValueError, match="overlap"):
        KinematicsReducer.reconstruct_joint_state(state, lock)


def test_reducer_rejects_retaining_dangling_collision_geometry():
    with pytest.raises(NotImplementedError, match="discarded links"):
        KinematicsReducer.reduce_dof(_branched_params(), ["tool"], remove_collision_spheres=False)


def test_cspace_and_limit_subsets_preserve_tensor_device():
    params = _branched_params()
    cspace = KinematicsReducer._create_cspace_subset(
        params.robot_cfg.cspace, ["tool_joint", "arm_joint"],
    )
    assert cspace.joint_names == ["tool_joint", "arm_joint"]
    limits = KinematicsReducer._create_joint_limits_subset(
        params.joint_limits, ["tool_joint", "arm_joint"]
    )
    assert limits.joint_names == ["tool_joint", "arm_joint"]
    assert limits.position.device.type == "cpu"

    tensor_cspace = CSpaceParams(
        ["arm_joint", "tool_joint", "collision_joint"],
        default_joint_position=torch.tensor([0.1, 0.2, 0.3]),
        cspace_distance_weight=torch.tensor([1.0, 2.0, 3.0]),
        null_space_weight=torch.tensor([3.0, 2.0, 1.0]),
    )
    reordered = KinematicsReducer._create_cspace_subset(
        tensor_cspace, ["tool_joint", "arm_joint"]
    )
    assert reordered.joint_names == ["tool_joint", "arm_joint"]
    torch.testing.assert_close(reordered.default_joint_position, torch.tensor([0.2, 0.1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_reduction_reconstruction_and_fk_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"), torch.float32)
    original = _branched_params(device_cfg)
    reduced = KinematicsReducer.reduce_dof(original, ["tool"])
    reduced_state = JointState.from_position(
        torch.zeros(2, 2, device="mps", requires_grad=True), joint_names=reduced.joint_names
    )
    full = KinematicsReducer.reconstruct_joint_state(
        reduced_state, reduced.lock_jointstate, original.joint_names
    )
    assert full.position.device.type == "mps"
    model = Kinematics(KinematicsCfg(device_cfg, ["tool"], reduced))
    out = model.compute_kinematics(reduced_state)
    assert out.tool_poses.position.device.type == "mps"
    torch.autograd.grad(out.tool_poses.position.sum(), reduced_state.position)
