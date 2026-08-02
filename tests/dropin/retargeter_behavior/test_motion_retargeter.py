"""Executable portable lifecycle checks for the pinned V2 retargeter facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose


def _criteria():
    return {"panda_hand": ToolPoseCriteria.track_position_and_orientation()}


def _config(*, device_cfg=DeviceCfg(), use_mpc=False, num_envs=1):
    return MotionRetargeterCfg.create(
        "franka.yml", _criteria(), num_envs=num_envs, use_mpc=use_mpc,
        self_collision_check=False, device_cfg=device_cfg,
        num_seeds_global=1, num_seeds_local=1, steps_per_target=1,
        mpc_warm_start_num_iters=1, mpc_cold_start_num_iters=1,
    )


def _goal_from_state(retargeter: MotionRetargeter, state: JointState) -> GoalToolPose:
    poses = retargeter.kinematics.compute_kinematics(state).tool_poses
    return GoalToolPose(
        retargeter.tool_frames, poses.position.unsqueeze(3), poses.quaternion.unsqueeze(3),
    )


def test_config_enforces_collision_loading_and_positive_capacities():
    no_collision = _config()
    assert no_collision.load_collision_spheres is False
    assert no_collision.tool_frames == ["panda_hand"]

    with pytest.raises(ValueError, match="load_collision_spheres"):
        MotionRetargeterCfg.create(
            "franka.yml", _criteria(), self_collision_check=True,
            load_collision_spheres=False,
        )
    with pytest.raises(ValueError, match="positive"):
        MotionRetargeterCfg.create("franka.yml", _criteria(), steps_per_target=0)
    with pytest.raises(ValueError, match="finite and positive"):
        MotionRetargeterCfg.create("franka.yml", _criteria(), optimization_dt=float("nan"))


def test_global_then_local_ik_and_sequence_have_v2_state_lifecycle():
    retargeter = MotionRetargeter(_config())
    goal = _goal_from_state(retargeter, retargeter.default_joint_state)

    first = retargeter.solve_frame(goal)
    assert first.joint_state.position.shape == (1, retargeter.action_dim)
    assert first.trajectory is None
    assert retargeter._prev_solution is not None
    assert retargeter._prev_velocity is None

    second = retargeter.solve_frame(goal)
    assert second.joint_state.position.shape == (1, retargeter.action_dim)
    assert second.trajectory is None
    assert retargeter._prev_velocity is not None
    assert retargeter._local_ik_solver.config.optimization_dt == retargeter.config.optimization_dt

    sequence = SequenceGoalToolPose(
        retargeter.tool_frames,
        goal.position.repeat(3, 1, 1, 1, 1),
        goal.quaternion.repeat(3, 1, 1, 1, 1),
    )
    output = retargeter.solve_sequence(sequence)
    assert output.joint_state.position.shape == (1, 3, retargeter.action_dim)
    assert output.joint_state.velocity.shape == (1, 3, retargeter.action_dim)
    assert output.trajectory is None


def test_batched_ik_retargeting_uses_configured_environment_capacity():
    retargeter = MotionRetargeter(_config(num_envs=2))
    state = JointState.from_position(
        retargeter.default_joint_state.position.repeat(2, 1), retargeter.joint_names
    )
    goal = _goal_from_state(retargeter, state)
    result = retargeter.solve_frame(goal)
    assert result.joint_state.position.shape == (2, retargeter.action_dim)
    assert retargeter._global_ik_solver.config.max_batch_size == 2


def test_mpc_retargeting_has_global_initialization_then_executed_endpoints():
    retargeter = MotionRetargeter(_config(use_mpc=True))
    goal = _goal_from_state(retargeter, retargeter.default_joint_state)

    first = retargeter.solve_frame(goal)
    assert first.trajectory is None
    assert retargeter._mpc_state is not None
    assert retargeter._mpc_solver.problem_batch_size == 1

    second = retargeter.solve_frame(goal)
    assert second.joint_state.position.shape == (1, retargeter.action_dim)
    assert second.trajectory is not None
    assert second.trajectory.position.shape == (1, 1, retargeter.action_dim)
    assert retargeter._prev_solution.shape == (1, retargeter.action_dim)


def test_batched_mpc_retargeting_preserves_each_environment_and_endpoint_stream():
    retargeter = MotionRetargeter(_config(use_mpc=True, num_envs=2))
    state = JointState.from_position(
        retargeter.default_joint_state.position.repeat(2, 1), retargeter.joint_names
    )
    goal = _goal_from_state(retargeter, state)
    first = retargeter.solve_frame(goal)
    second = retargeter.solve_frame(goal)

    assert first.joint_state.position.shape == (2, retargeter.action_dim)
    assert second.joint_state.position.shape == (2, retargeter.action_dim)
    assert second.trajectory is not None
    assert second.trajectory.position.shape == (2, 1, retargeter.action_dim)
    assert torch.allclose(second.joint_state.position, second.trajectory.position[:, -1])
    assert retargeter._mpc_solver._trajopt.config.max_batch_size == 2


def test_retargeter_rejects_multi_frame_goal_at_frame_api_boundary():
    retargeter = MotionRetargeter(_config())
    goal = _goal_from_state(retargeter, retargeter.default_joint_state)
    multi_frame = GoalToolPose(
        goal.tool_frames, goal.position.repeat(1, 2, 1, 1, 1),
        goal.quaternion.repeat(1, 2, 1, 1, 1),
    )
    with pytest.raises(ValueError, match="exactly one target frame"):
        retargeter.solve_frame(multi_frame)


def test_mpc_multi_tool_boundary_and_goal_type_errors_are_explicit():
    criteria = {
        "panda_hand": ToolPoseCriteria.track_position(),
        "panda_link7": ToolPoseCriteria.track_position(),
    }
    with pytest.raises(NotImplementedError, match="one tracked tool frame"):
        MotionRetargeter(MotionRetargeterCfg.create(
            "franka.yml", criteria, use_mpc=True, self_collision_check=False,
        ))
    retargeter = MotionRetargeter(_config())
    with pytest.raises(TypeError, match="GoalToolPose"):
        retargeter.solve_frame(torch.zeros(1, 7))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_retargeter_runs_with_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"), torch.float32)
    retargeter = MotionRetargeter(_config(device_cfg=device))
    goal = _goal_from_state(retargeter, retargeter.default_joint_state)
    first = retargeter.solve_frame(goal)
    second = retargeter.solve_frame(goal)
    assert first.joint_state.device.type == "mps"
    assert second.joint_state.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_batched_mpc_retargeter_runs_with_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"), torch.float32)
    retargeter = MotionRetargeter(_config(device_cfg=device, use_mpc=True, num_envs=2))
    state = JointState.from_position(
        retargeter.default_joint_state.position.repeat(2, 1), retargeter.joint_names
    )
    goal = _goal_from_state(retargeter, state)
    retargeter.solve_frame(goal)
    output = retargeter.solve_frame(goal)
    assert output.joint_state.device.type == "mps"
    assert output.trajectory is not None
    assert output.trajectory.device.type == "mps"
