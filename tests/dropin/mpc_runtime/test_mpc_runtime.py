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
