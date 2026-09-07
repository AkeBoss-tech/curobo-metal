"""Public-facade behavior and signature checks independent of private imports."""

from __future__ import annotations

import inspect

import pytest
from curobo.types import DeviceCfg
import torch

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.types import GoalToolPose, JointState


def test_public_trajectory_pose_signature_and_ik_composition():
    assert list(inspect.signature(TrajectoryOptimizer.solve_pose).parameters) == [
        "self", "goal_tool_poses", "current_state", "seed_config", "seed_traj",
        "return_seeds", "num_seeds", "dt", "use_implicit_goal",
        "finetune_attempts", "goal_state", "initial_iters", "time_optimal_iters",
        "finetune_iters", "finetune_dt_scale",
    ]
    cfg = TrajectoryOptimizerCfg.create(
        "franka.yml", num_seeds=2, override_optimizer_num_iters={"lbfgs": 2},
        use_cuda_graph=True,
        device_cfg=DeviceCfg("cpu"),
    )
    solver = TrajectoryOptimizer(cfg)
    current = solver.default_joint_state
    pose = solver.compute_kinematics(current).tool_poses.get_link_pose("panda_hand")
    goal = GoalToolPose.from_poses({"panda_hand": pose}, ordered_tool_frames=["panda_hand"])
    result = solver.solve_pose(
        goal, current, seed_config=current.position.reshape(1, 1, -1), initial_iters=2
    )
    assert bool(result.success.all().item())
    assert "ik_result" in result.debug_info
    assert result.js_solution.position.shape[:2] == (1, 1)


def test_public_collision_names_match_pinned_signature_and_execute():
    assert RobotCollisionChecker.__name__ == "RobotSceneCollision"
    assert list(inspect.signature(RobotCollisionChecker.sample).parameters) == [
        "self", "n", "mask_valid", "env_query_idx"
    ]
    assert list(inspect.signature(RobotCollisionChecker.get_kinematics).parameters) == [
        "self", "joint_position"
    ]
    assert list(inspect.signature(RobotCollisionChecker.validate).parameters) == [
        "self", "q", "env_query_idx"
    ]
    cfg = RobotCollisionCheckerCfg.load_from_config("franka.yml", device_cfg=DeviceCfg("cpu"))
    checker = RobotCollisionChecker(cfg)
    q = checker.kinematics.default_joint_state.position.reshape(1, 1, -1)
    assert checker.validate(q).shape == (1, 1)
    assert checker.sample(2, mask_valid=False).shape == (2, checker.kinematics.dof)


def test_public_kinematics_mesh_call_has_precise_portable_boundary():
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg("cpu")))
    assert list(inspect.signature(Kinematics.get_robot_as_mesh).parameters) == [
        "self", "joint_position"
    ]
    with pytest.raises(NotImplementedError, match="mesh assets"):
        kin.get_robot_as_mesh(torch.zeros(1, kin.dof))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_public_trajectory_pose_route_runs_without_mps_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo.types import DeviceCfg

    cfg = TrajectoryOptimizerCfg.create(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps")), num_seeds=2,
        override_optimizer_num_iters={"lbfgs": 2},
    )
    solver = TrajectoryOptimizer(cfg)
    current = solver.default_joint_state
    pose = solver.compute_kinematics(current).tool_poses.get_link_pose("panda_hand")
    goal = GoalToolPose.from_poses({"panda_hand": pose}, ordered_tool_frames=["panda_hand"])
    result = solver.solve_pose(goal, current, initial_iters=2)
    assert bool(result.success.all().item())
    assert result.solution.device.type == "mps"
