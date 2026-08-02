"""Portable behavioral coverage for the single-problem MotionPlanner facade."""

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState


def _planner():
    cfg = MotionPlannerCfg.create(
        "franka.yml", num_ik_seeds=2, num_trajopt_seeds=1,
        use_cuda_graph=False,
    )
    # Keep this deterministic graph smoke small; its output is an ordinary
    # PyTorch trajectory seed, not a CUDA graph capture.
    cfg.graph_planner_config = {"max_nodes": 16, "new_nodes_per_iteration": 4}
    cfg.trajopt_solver_config.max_iterations = 2
    return MotionPlanner(cfg)


def test_prm_seed_lifecycle_link_collision_and_criteria():
    planner = _planner()
    assert planner.graph_planner is not None
    current = planner.default_joint_state
    seeds = (current.position + 0.002).reshape(1, 1, -1)
    graph_seed = planner._get_graph_seed_trajectories(current, seeds)
    assert graph_seed is not None
    assert graph_seed.shape[-2:] == (
        planner.trajopt_solver.action_horizon, planner.action_dim
    )

    params = planner.kinematics.config.kinematics_config
    before = params.get_link_spheres("panda_link0")[..., 3].clone()
    planner.disable_link_collision(["panda_link0"])
    assert torch.all(params.get_link_spheres("panda_link0")[..., 3] <= 0)
    planner.enable_link_collision(["panda_link0"])
    torch.testing.assert_close(params.get_link_spheres("panda_link0")[..., 3], before.abs())

    criteria = {"panda_hand": ToolPoseCriteria.track_position()}
    planner.update_tool_pose_criteria(criteria)
    assert planner.ik_solver.config.tool_pose_criteria == criteria
    assert planner.trajopt_solver.config.tool_pose_criteria == criteria
    planner.reset_seed()


def test_motion_planner_rejects_invalid_state_ranks_and_warmup_index():
    planner = _planner()
    current = planner.default_joint_state
    goal = JointState.from_position(current.position + 0.001, planner.joint_names)
    invalid = JointState.from_position(
        current.position.reshape(1, 1, -1), planner.joint_names
    )
    with pytest.raises(ValueError, match="current_state"):
        planner.plan_cspace(goal, invalid)
    with pytest.raises(ValueError, match="warmup_joint_index"):
        planner.warmup(warmup_joint_index=planner.action_dim)
