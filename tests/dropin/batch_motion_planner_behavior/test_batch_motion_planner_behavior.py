"""Behavioral coverage for the portable pinned BatchMotionPlanner contract."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _planner(*, batch_size=2, multi_env=False, device_cfg=DeviceCfg()):
    cfg = MotionPlannerCfg.create(
        "franka.yml", max_batch_size=batch_size, multi_env=multi_env,
        num_ik_seeds=2, num_trajopt_seeds=1, device_cfg=device_cfg,
    )
    cfg.trajopt_solver_config.max_iterations = 2
    return BatchMotionPlanner(cfg)


def _trajectory_result(success, marker, names):
    success = torch.tensor(success, dtype=torch.bool)
    batch = success.shape[0]
    position = torch.full((batch, 1, 3, len(names)), float(marker))
    solution = position.clone()
    return TrajOptSolverResult(
        success=success,
        solution=solution,
        js_solution=JointState.from_position(position, names),
        cspace_error=torch.full((batch, 1), float(marker)),
        total_time=float(marker),
        solve_time=float(marker),
    )


def test_batch_cspace_keeps_first_success_result_rows(monkeypatch):
    planner = _planner()
    names = planner.joint_names
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), names)
    goal = JointState.from_position(start.position + 0.01, names)
    attempts = iter((
        _trajectory_result([[True], [False]], 1, names),
        _trajectory_result([[False], [True]], 2, names),
    ))
    monkeypatch.setattr(planner.trajopt_solver, "solve_cspace", lambda *args, **kwargs: next(attempts))
    monkeypatch.setattr(planner, "_finish_trajectory", lambda value: value)

    result = planner.plan_cspace(goal, start, max_attempts=2, success_ratio=1.0)

    assert result.success.tolist() == [[True], [True]]
    assert result.js_solution.position[:, 0, 0, 0].tolist() == [1.0, 2.0]
    assert result.solution[:, 0, 0, 0].tolist() == [1.0, 2.0]
    assert result.total_time == 3.0


def test_batch_pose_forwards_trajectory_options_and_goalset(monkeypatch):
    planner = _planner()
    names = planner.joint_names
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), names)
    position = torch.zeros((2, 1, len(planner.tool_frames), 1, 3))
    quaternion = torch.zeros((2, 1, len(planner.tool_frames), 1, 4))
    quaternion[..., 0] = 1.0
    goal = GoalToolPose(planner.tool_frames, position, quaternion)
    ik = SimpleNamespace(
        success=torch.ones((2, 1), dtype=torch.bool), total_time=0.25,
        solution=start.position[:, None], goalset_index=torch.zeros((2, 1, 1), dtype=torch.long),
    )
    monkeypatch.setattr(planner.ik_solver, "solve_pose", lambda *args, **kwargs: ik)
    observed = {}

    def solve_pose(*args, **kwargs):
        observed.update(kwargs)
        return _trajectory_result([[True], [True]], 1, names)

    monkeypatch.setattr(planner.trajopt_solver, "solve_pose", solve_pose)
    monkeypatch.setattr(planner, "_finish_trajectory", lambda value: value)
    result = planner.plan_pose(
        goal, start, use_implicit_goal=False, max_attempts=1,
        finetune_attempts=3, initial_iters=7, time_optimal_iters=5,
        finetune_iters=4, finetune_dt_scale=0.7,
    )

    assert result.goalset_index.shape == (2, 1, 1)
    assert observed["use_implicit_goal"] is False
    assert observed["finetune_attempts"] == 3
    assert observed["initial_iters"] == 7
    assert observed["time_optimal_iters"] == 5
    assert observed["finetune_iters"] == 4
    assert observed["finetune_dt_scale"] == 0.7


def test_batch_pose_executes_a_real_two_problem_fk_goal():
    planner = _planner()
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), planner.joint_names)
    tool_poses = planner.compute_kinematics(start).tool_poses
    goal = GoalToolPose(
        planner.tool_frames, tool_poses.position.unsqueeze(3), tool_poses.quaternion.unsqueeze(3)
    )
    result = planner.plan_pose(goal, start, max_attempts=1)
    assert result.success.tolist() == [[True], [True]]
    assert result.js_solution.position.shape[:2] == (2, 1)


def test_batch_grasp_approach_only_preserves_per_problem_result_shape():
    planner = _planner()
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), planner.joint_names)
    tool_poses = planner.compute_kinematics(start).tool_poses
    grasps = GoalToolPose(
        planner.tool_frames, tool_poses.position.unsqueeze(3), tool_poses.quaternion.unsqueeze(3)
    )
    result = planner.plan_grasp(
        grasps, start, grasp_approach_offset=0.0,
        plan_approach_to_grasp=False, plan_grasp_to_lift=False,
    )
    assert result.success.shape == (2,)
    assert result.approach_success.shape == (2,)
    assert result.approach_trajectory.position.shape[:2] == (2, 1)


def test_graph_seeds_flatten_and_restore_batch_seed_shape():
    planner = _planner(batch_size=2)
    names = planner.joint_names
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), names)
    goals = start.position[:, None].expand(-1, 3, -1).clone()
    observed = {}

    def find_path(starts, endpoints, **kwargs):
        observed["starts"] = starts
        observed["endpoints"] = endpoints
        values = torch.arange(2 * 3 * 4 * len(names), dtype=starts.dtype).reshape(6, 4, len(names))
        return SimpleNamespace(success=torch.tensor([True, False, True, False, True, True]),
                               interpolated_waypoints=values)

    planner.graph_planner = SimpleNamespace(find_path=find_path)
    seeds = planner._get_graph_seed_trajectories(start, goals)
    assert observed["starts"].shape == (6, 7)
    assert observed["endpoints"].shape == (6, 7)
    assert seeds.shape == (2, 3, 4, 7)
    assert seeds[1, 2, 3, 6].item() == 167


def test_multi_environment_disables_shared_graph_and_rejects_invalid_ratios():
    planner = _planner(multi_env=True)
    assert planner.graph_planner is None
    start = planner.default_joint_state.unsqueeze(0)
    goal = JointState.from_position(start.position + 0.01, planner.joint_names)
    with pytest.raises(ValueError, match="success_ratio"):
        planner.plan_cspace(goal, start, success_ratio=1.1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_batch_cspace_runs_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    planner = _planner(device_cfg=DeviceCfg(torch.device("mps"), torch.float32))
    start = JointState.from_position(planner.default_joint_state.position.repeat(2, 1), planner.joint_names)
    goal = JointState.from_position(start.position + 0.002, planner.joint_names)
    result = planner.plan_cspace(goal, start, max_attempts=1)
    assert bool(result.success.all().item())
    assert result.js_solution.position.device.type == "mps"
