"""Shared executable checks used by the packaged getting-started examples."""

from __future__ import annotations

import torch

from curobo.content import get_robot_configs_path
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.robot_builder import RobotBuilder
from curobo.types import DeviceCfg, GoalToolPose, JointState


def accelerator() -> DeviceCfg:
    return DeviceCfg(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))


def forward_kinematics() -> None:
    robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=accelerator()))
    q = robot.default_joint_state.position.reshape(1, -1).clone().requires_grad_()
    state = robot.compute_kinematics(JointState.from_position(q, robot.joint_names))
    state.tool_poses.position.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def inverse_kinematics() -> None:
    solver = InverseKinematics(
        InverseKinematicsCfg.create("franka.yml", device_cfg=accelerator(), num_seeds=2, use_cuda_graph=False)
    )
    current = solver.default_joint_state
    pose = solver.compute_kinematics(current).tool_poses.get_link_pose(solver.tool_frames[-1])
    goal = GoalToolPose.from_poses({solver.tool_frames[-1]: pose}, ordered_tool_frames=solver.tool_frames)
    result = solver.solve_pose(goal, current_state=current)
    assert bool(result.success.all().item())


def motion_planning() -> None:
    cfg = MotionPlannerCfg.create(
        "franka.yml", device_cfg=accelerator(), num_ik_seeds=2, num_trajopt_seeds=1,
        use_cuda_graph=False,
    )
    cfg.trajopt_solver_config.max_iterations = 2
    planner = MotionPlanner(cfg)
    current = planner.default_joint_state
    pose = planner.compute_kinematics(current).tool_poses.get_link_pose(planner.tool_frames[-1])
    goal = GoalToolPose.from_poses({planner.tool_frames[-1]: pose}, ordered_tool_frames=planner.tool_frames)
    result = planner.plan_pose(goal, current, max_attempts=1)
    assert bool(result.success.all().item())


def reactive_control() -> None:
    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", device_cfg=accelerator(), warm_start_optimization_num_iters=1,
        cold_start_optimization_num_iters=1, use_cuda_graph=False,
    )
    solver = ModelPredictiveControl(cfg)
    current = solver.default_joint_state
    assert solver.setup(current)
    goal = current.clone()
    goal.position = goal.position + 0.01
    assert solver.update_goal_state(goal)
    result = solver.optimize_next_action(current)
    assert result.next_action.position.shape[-1] == solver.action_dim


def build_robot_model() -> None:
    source = get_robot_configs_path() / "franka.yml"
    builder = RobotBuilder.from_config(str(source))
    config = builder.build()
    assert config is not None
    assert builder.tool_frames == ["panda_hand"]
    assert builder.collision_link_names
