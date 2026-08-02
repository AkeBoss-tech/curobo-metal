"""Portable lifecycle coverage for solver, motion, and rollout compatibility."""

import pytest
import torch

from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


def _trajopt():
    cfg = MotionPlannerCfg.create(
        "franka.yml", num_trajopt_seeds=2, use_cuda_graph=False
    ).trajopt_solver_config
    cfg.max_iterations = 2
    return TrajOptSolver(cfg)


def test_trajopt_pose_goal_state_route_and_cuda_boundary():
    solver = _trajopt()
    current = solver.default_joint_state
    goal = JointState.from_position(current.position + 0.002, solver.joint_names)
    # ``solve_pose`` now executes the native c-space route when a caller has a
    # known joint endpoint, rather than exposing a non-upstream-only stub.
    result = solver.solve_pose(None, current, goal_state=goal, initial_iters=2)
    assert isinstance(result, TrajOptSolverResult)
    assert result.js_solution.position.shape[:2] == (1, 1)
    tool_pose = solver.compute_kinematics(current).tool_poses.get_link_pose(
        solver.tool_frames[-1]
    )
    pose_goal = GoalToolPose.from_poses(
        {solver.tool_frames[-1]: tool_pose}, ordered_tool_frames=[solver.tool_frames[-1]]
    )
    composed = solver.solve_pose(pose_goal, current, initial_iters=2)
    assert "ik_result" in composed.debug_info
    assert torch.isfinite(composed.solution).all()
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        solver.reset_cuda_graph()


def test_planner_scene_mutation_and_attachment_lifecycle():
    cfg = MotionPlannerCfg.create("franka.yml", use_cuda_graph=False)
    cfg.trajopt_solver_config.max_iterations = 2
    planner = MotionPlanner(cfg)
    scene = SceneCfg(sphere=[Sphere("keepout", position=[3.0, 0.0, 0.0], radius=0.1)])
    planner.update_world(scene)
    assert planner.attachment_manager is not None
    assert planner.ik_solver.scene_collision_checker is planner._scene_collision
    planner.clear_scene_cache()
    assert planner.warmup(num_warmup_iterations=1)


def test_solver_core_goal_lifecycle_and_ik_cuda_boundary():
    cfg = IKSolverCfg.create("franka.yml", num_seeds=2, use_cuda_graph=False)
    core = SolverCore(cfg.core_cfg)
    state = core.default_joint_state.unsqueeze(0)
    solve_state = SolveState(SolveMode.SINGLE, 1, 1, num_seeds=2)
    goal = core.prepare_goal_buffer(solve_state, None, current_state=state, goal_state=state)
    assert goal.goal_js.position.shape == (1, 7)
    assert goal.current_js.position.shape == (1, 7)

    ik = IKSolver(cfg)
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        ik.reset_cuda_graph()


def test_trajopt_topk_result_preserves_seed_payloads():
    success = torch.tensor([[True, True, False]])
    solution = torch.arange(6.0).reshape(1, 3, 2)
    result = TrajOptSolverResult(
        success=success,
        solution=solution,
        seed_cost=torch.tensor([[3.0, 1.0, 2.0]]),
        optimized_seeds=solution[:, :, None],
        batch_size=1,
        num_seeds=3,
    )
    selected = result.get_topk_seeds(2)
    assert selected.seed_cost.tolist() == [[1.0, 2.0]]
    assert selected.solution.tolist() == [[[2.0, 3.0], [4.0, 5.0]]]
    assert selected.num_seeds == 2
