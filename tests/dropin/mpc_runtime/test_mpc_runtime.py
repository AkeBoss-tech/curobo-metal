import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo.model_predictive_control import (
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)


def _mpc():
    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", warm_start_optimization_num_iters=1,
        cold_start_optimization_num_iters=1,
    )
    solver = ModelPredictiveControl(cfg)
    current = solver.default_joint_state
    assert solver.setup(current)
    return solver, current


def test_pose_goal_scene_lifecycle_and_seed_validation():
    mpc, current = _mpc()
    tool_pose = mpc.compute_kinematics(current.unsqueeze(0)).tool_poses
    goal = mpc._as_goal_tool_pose(tool_pose, mpc.solve_state.tool_frames)
    assert mpc.update_goal_tool_poses(goal, use_best_effort_ik=True)

    world = SceneCfg(cuboid=[Cuboid(
        name="table", pose=[0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
        dims=[2.0, 2.0, 0.1],
    )])
    mpc.update_world(world)
    assert mpc.scene_collision_checker.check_obstacle_exists("table")
    assert mpc._ik.scene_collision_checker is mpc.scene_collision_checker

    with pytest.raises(ValueError, match="seed_trajectory"):
        mpc.update_seed_trajectory(torch.zeros(1, 2, mpc.action_dim))
    mpc.update_seed_trajectory(torch.zeros(1, mpc.action_horizon, mpc.action_dim))
    assert mpc.debug_dump()["warm_start"]


def test_rejects_time_varying_pose_goals():
    mpc, current = _mpc()
    tool_pose = mpc.compute_kinematics(current.unsqueeze(0)).tool_poses
    goal = mpc._as_goal_tool_pose(tool_pose, mpc.solve_state.tool_frames)
    time_varying = type(goal)(
        goal.tool_frames, goal.position.repeat(1, 2, 1, 1, 1),
        goal.quaternion.repeat(1, 2, 1, 1, 1),
    )
    with pytest.raises(NotImplementedError, match="single-time"):
        mpc.update_goal_tool_poses(time_varying)


def test_next_action_consumes_portable_receding_horizon_buffer():
    mpc, current = _mpc()
    goal = current.clone()
    goal.position = goal.position + 0.02
    assert mpc.update_goal_state(goal)

    first = mpc.optimize_next_action(current)
    solved_after_first = mpc.debug_dump()["solve_count"]
    second = mpc.optimize_next_action(current)
    state = mpc.debug_dump()

    assert solved_after_first == 1
    assert state["solve_count"] == 1
    assert first.metrics["command_index"] == 0
    assert second.metrics["command_index"] == 1
    assert first.next_action.position.shape == (1, mpc.action_dim)
    assert not first.metrics["reoptimized"]
    assert not second.metrics["reoptimized"]


def test_safe_deceleration_uses_velocity_and_supports_mixed_batch():
    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", max_batch_size=2, warm_start_optimization_num_iters=1,
        cold_start_optimization_num_iters=1,
    )
    mpc = ModelPredictiveControl(cfg)
    position = mpc.default_joint_state.position.repeat(2, 1)
    current = type(mpc.default_joint_state)(
        position, torch.full_like(position, 0.2), torch.zeros_like(position),
        mpc.joint_names,
    )
    assert mpc.setup(current)
    plan = mpc.prepare_safe_deceleration_trajectory(
        current, torch.tensor([True, False]), deceleration_profile="linear"
    )
    assert plan.shape == (2, mpc.action_horizon, mpc.action_dim)
    assert torch.all(plan[0, 1] > plan[0, 0])
    torch.testing.assert_close(plan[1], current.position[1].expand_as(plan[1]))
    with pytest.raises(ValueError, match="deceleration_profile"):
        mpc.prepare_safe_deceleration_trajectory(
            current, torch.tensor([True, False]), deceleration_profile="bad"
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mpc_buffer_and_deceleration_stay_on_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"), torch.float32),
        warm_start_optimization_num_iters=1, cold_start_optimization_num_iters=1,
    )
    mpc = ModelPredictiveControl(cfg)
    current = mpc.default_joint_state
    assert mpc.setup(current)
    result = mpc.optimize_next_action(current)
    assert result.next_action.position.device.type == "mps"
    safe = mpc.prepare_safe_deceleration_trajectory(
        current, torch.tensor([True], device="mps")
    )
    assert safe.device.type == "mps"
