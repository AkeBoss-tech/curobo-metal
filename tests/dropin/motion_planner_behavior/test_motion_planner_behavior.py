"""Portable behavioral coverage for the single-problem MotionPlanner facade."""

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


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


def test_pose_route_uses_portable_ik_composition_and_preserves_lifecycle():
    """The public planner must not route through CUDA-only implicit goals."""
    planner = _planner()
    state = planner.default_joint_state
    pose = planner.compute_kinematics(state).tool_poses.get_link_pose(
        planner.tool_frames[-1]
    )
    goal = GoalToolPose.from_poses(
        {planner.tool_frames[-1]: pose}, ordered_tool_frames=planner.tool_frames
    )
    result = planner.plan_pose(goal, state, max_attempts=1)
    assert bool(result.success.all().item())
    assert "ik_result" in result.debug_info
    assert result.js_solution.position.shape[:2] == (1, 1)

    # V2's goalset warmup layout is useful even though CUDA graph capture is
    # unavailable: every frame receives exactly one pose per requested goal.
    warmup_poses = planner._make_warmup_goalset(state.unsqueeze(0), 2, 0, 0.01)
    warmup_goal = GoalToolPose.from_poses(
        warmup_poses, ordered_tool_frames=planner.tool_frames, num_goalset=2
    )
    assert warmup_goal.position.shape == (1, 1, len(planner.tool_frames), 2, 3)

    planner.destroy()
    planner.destroy()  # lifecycle destruction is intentionally idempotent


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_pose_route_is_device_resident_without_mps_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = MotionPlannerCfg.create(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"), torch.float32),
        num_ik_seeds=2, num_trajopt_seeds=1, use_cuda_graph=False,
    )
    cfg.trajopt_solver_config.max_iterations = 2
    planner = MotionPlanner(cfg)
    state = planner.default_joint_state
    pose = planner.compute_kinematics(state).tool_poses.get_link_pose(
        planner.tool_frames[-1]
    )
    goal = GoalToolPose.from_poses(
        {planner.tool_frames[-1]: pose}, ordered_tool_frames=planner.tool_frames
    )
    result = planner.plan_pose(goal, state, max_attempts=1)
    assert bool(result.success.all().item())
    assert result.js_solution.position.device.type == "mps"
