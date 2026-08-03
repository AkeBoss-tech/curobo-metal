"""High-value lifecycle coverage for portable IK and seeded IK composition."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solver(**kwargs) -> IKSolver:
    kwargs.setdefault("num_seeds", 2)
    return IKSolver(IKSolverCfg.create("franka.yml", use_cuda_graph=False, **kwargs))


def _exact_goal(solver: IKSolver) -> GoalToolPose:
    state = solver.compute_kinematics(solver.default_joint_state)
    return GoalToolPose(
        solver.tool_frames,
        state.tool_poses.position.unsqueeze(3),
        state.tool_poses.quaternion.unsqueeze(3),
    )


def test_max_batch_padding_is_used_internally_and_hidden_from_result():
    solver = _solver(max_batch_size=3)
    result = solver.solve_pose(
        _exact_goal(solver), current_state=solver.default_joint_state, run_optimizer=False
    )

    assert result.batch_size == result.success.shape[0] == 1
    assert result.metrics["world_clearance"].shape == (1, 1)
    assert solver.solve_state.batch_size == 3
    assert solver.goal_registry_manager.goal_buffer.link_goal_poses.batch_size == 3
    assert solver._last_result is result


def test_active_self_collision_cost_is_a_feasibility_constraint():
    solver = _solver(self_collision_check=False)

    class _AlwaysColliding:
        def __call__(self, spheres):
            return spheres.new_ones((*spheres.shape[:2], 1))

    # A compact synthetic collision cost isolates result semantics from the
    # Franka fixture's intentionally non-colliding default posture.
    solver._self_collision_cost = _AlwaysColliding()
    result = solver.solve_pose(
        _exact_goal(solver), current_state=solver.default_joint_state, run_optimizer=False
    )

    assert not result.feasible.any()
    assert not result.success.any()
    assert torch.equal(result.metrics["self_collision_cost"], torch.ones_like(result.seed_cost))


def test_seed_count_boundary_lm_budget_and_typed_world_cfg_are_honored():
    solver = _solver(num_seeds=40, seed_solver_num_seeds=2)
    assert solver.seed_ik_solver is not None
    # Pinned V2 doubles the LM budget when the main optimizer wants more
    # candidates than its nominal seed-solver capacity.
    assert solver.seed_ik_solver.config.num_seeds == 80
    with pytest.raises(ValueError, match="more seeds"):
        solver.solve_pose(
            _exact_goal(solver),
            seed_config=solver.default_joint_position.reshape(1, 1, -1).expand(1, 41, -1),
            run_optimizer=False,
        )
    with pytest.raises(ValueError, match="finite"):
        solver.solve_pose(
            _exact_goal(solver),
            seed_config=torch.full((1, 1, solver.action_dim), float("nan")),
            run_optimizer=False,
        )

    config = SceneCollisionCfg(device_cfg=solver.device_cfg)
    solver.update_world(config)
    assert isinstance(solver.scene_collision_checker, SceneCollision)
