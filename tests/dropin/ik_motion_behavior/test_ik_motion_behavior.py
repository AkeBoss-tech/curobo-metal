"""Behavioral coverage for the portable high-level IK facade."""

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solver_ik import IKSolver, _pad_batch_inputs, _slice_batch_result
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solver(**kwargs) -> IKSolver:
    kwargs.setdefault("num_seeds", 2)
    return IKSolver(IKSolverCfg.create(
        "franka.yml", use_cuda_graph=False, **kwargs
    ))


def _two_goal_pose(solver: IKSolver, exact_index: int) -> GoalToolPose:
    state = solver.compute_kinematics(solver.default_joint_state)
    position = state.tool_poses.position[:, 0]
    quaternion = state.tool_poses.quaternion[:, 0]
    goals_position = position[:, :, None, :].expand(-1, -1, 2, -1).clone()
    goals_quaternion = quaternion[:, :, None, :].expand(-1, -1, 2, -1).clone()
    goals_position[:, :, 1 - exact_index, 0] += 0.3
    return GoalToolPose(
        solver.tool_frames,
        goals_position.unsqueeze(1),
        goals_quaternion.unsqueeze(1),
    )


def test_metrics_only_goalset_selects_the_lowest_cost_candidate():
    solver = _solver(max_goalset=2, optimization_dt=0.1)
    current = solver.default_joint_state
    goal = _two_goal_pose(solver, exact_index=1)

    result = solver.solve_pose(goal, current_state=current, run_optimizer=False)

    assert result.success.tolist() == [[True]]
    assert result.goalset_index.tolist() == [[[1]]]
    torch.testing.assert_close(result.solution[0, 0], current.position)
    torch.testing.assert_close(result.js_solution.velocity, torch.zeros_like(result.solution))
    assert result.metrics["position_error_per_link"].shape == (1, 1, 1)
    assert result.metrics["world_clearance"].isinf().all()


def test_ik_goalset_and_seed_shape_boundaries_are_explicit():
    solver = _solver(max_goalset=1)
    with pytest.raises(ValueError, match="max_goalset"):
        solver.solve_pose(_two_goal_pose(solver, exact_index=0), run_optimizer=False)

    goal = _two_goal_pose(_solver(max_goalset=2), exact_index=0)
    with pytest.raises(ValueError, match="seed_config batch"):
        _solver(max_goalset=2).solve_pose(
            goal,
            seed_config=torch.zeros((2, 7)),
            run_optimizer=False,
        )


def test_ik_world_update_mutates_the_existing_scene_collision_adapter():
    scene = SceneCollision(SceneCollisionCfg(
        scene_model=SceneCfg(sphere=[Sphere("old", position=[3.0, 0.0, 0.0], radius=0.1)]),
    ))
    solver = _solver()
    solver._scene_collision_checker = scene
    replacement = SceneCfg(sphere=[Sphere("new", position=[2.0, 0.0, 0.0], radius=0.1)])

    solver.update_world(replacement)

    assert solver.scene_collision_checker is scene
    assert scene.check_obstacle_exists("new")
    assert not scene.check_obstacle_exists("old")
    with pytest.raises(NotImplementedError, match="YAML/USD"):
        solver.update_world("world.yml")


def test_ik_persists_typed_goal_lifecycle_and_uses_portable_lm_seeding():
    solver = _solver(max_goalset=2, num_seeds=2)
    current = solver.default_joint_state
    result = solver.solve_pose(_two_goal_pose(solver, exact_index=0), current_state=current)

    assert result.success.tolist() == [[True]]
    assert result.metrics["backend"] == "torch-adam+lm"
    assert result.metrics["iterations"] >= 1
    assert solver.seed_ik_solver is not None
    assert solver.solve_state.solve_type is SolveMode.SINGLE
    assert solver.solve_state.num_goalset == 2
    assert solver.goal_registry_manager.goal_buffer.link_goal_poses is not None


def test_ik_batch_padding_helpers_clone_and_restore_result_ranks():
    solver = _solver(max_batch_size=2, max_goalset=2)
    current = solver.default_joint_state
    current = current.from_position(current.position.unsqueeze(0), solver.joint_names)
    goal = _two_goal_pose(solver, exact_index=0)
    padded_goal, padded_state, padded_seed = _pad_batch_inputs(
        goal, current, current.position.reshape(1, 1, -1), 1, 2
    )
    assert padded_goal.batch_size == padded_state.position.shape[0] == padded_seed.shape[0] == 2
    padded_goal.position[1, 0, 0, 0, 0] += 1.0
    assert not torch.equal(padded_goal.position[0], padded_goal.position[1])

    result = solver.solve_pose(goal, current_state=current, run_optimizer=False)
    result.solution = torch.cat((result.solution, result.solution), dim=0)
    result.success = torch.cat((result.success, result.success), dim=0)
    result.js_solution.position = torch.cat((result.js_solution.position, result.js_solution.position), dim=0)
    _slice_batch_result(result, 1)
    assert result.solution.shape[0] == result.js_solution.position.shape[0] == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_ik_lm_seed_and_goal_registry_stay_on_mps_without_fallback():
    from curobo._src.types.device_cfg import DeviceCfg

    solver = IKSolver(IKSolverCfg.create(
        "franka.yml", num_seeds=2, use_cuda_graph=False,
        device_cfg=DeviceCfg(torch.device("mps"), torch.float32),
    ))
    state = solver.compute_kinematics(solver.default_joint_state)
    goal = GoalToolPose(
        solver.tool_frames, state.tool_poses.position.unsqueeze(3),
        state.tool_poses.quaternion.unsqueeze(3),
    )
    result = solver.solve_pose(goal, current_state=solver.default_joint_state)
    assert result.solution.device.type == "mps"
    assert solver.goal_registry_manager.goal_buffer.link_goal_poses.device.type == "mps"
