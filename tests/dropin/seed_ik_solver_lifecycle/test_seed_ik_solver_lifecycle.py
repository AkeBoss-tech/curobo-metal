"""Lifecycle tests for the portable seeded Levenberg--Marquardt solver."""

from __future__ import annotations

import pytest
import torch

from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.types.tool_pose import GoalToolPose


def _cfg(device_cfg: DeviceCfg = DeviceCfg("cpu"), **kwargs) -> SeedIKSolverCfg:
    return SeedIKSolverCfg.create(
        "franka.yml",
        device_cfg=device_cfg,
        num_seeds=2,
        max_iterations=16,
        inner_iterations=4,
        position_tolerance=0.01,
        orientation_tolerance=0.1,
        **kwargs,
    )


def _goal(solver: SeedIKSolver) -> GoalToolPose:
    state = solver.compute_kinematics(solver.default_joint_position)
    pose = state.tool_poses.get_link_pose("panda_hand")
    return GoalToolPose.from_poses({"panda_hand": pose}, ordered_tool_frames=["panda_hand"])


def test_cfg_create_preserves_robotcfg_and_accepts_robot_mapping():
    from_name = _cfg(device_cfg=DeviceCfg("cpu"))
    from_cfg = SeedIKSolverCfg.create(from_name.robot_config, num_seeds=3, device_cfg=DeviceCfg("cpu"))
    assert from_cfg.robot_config is from_name.robot_config
    assert from_cfg.num_seeds == 3

    mapping = from_name.robot_config.kinematics.to_mapping()
    from_mapping = SeedIKSolverCfg.create(mapping, num_seeds=1, device_cfg=DeviceCfg("cpu"))
    assert isinstance(from_mapping.robot_config, RobotCfg)
    assert from_mapping.robot_config.cspace.joint_names == from_name.robot_config.cspace.joint_names


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"joint_limit_margin": 0.5}, "joint_limit_margin"),
        ({"lambda_initial": float("nan")}, "lambda_initial"),
        ({"lambda_initial": 1.0e11}, "lambda_initial"),
        ({"max_problems_mini_batch": 0}, "max_problems_mini_batch"),
        ({"sampler_seed": -1}, "sampler_seed"),
    ],
)
def test_cfg_rejects_invalid_portable_runtime_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _cfg(**kwargs)


def test_destroy_then_resolve_recreates_shape_dependent_buffers():
    solver = SeedIKSolver(_cfg(device_cfg=DeviceCfg("cpu")))
    goal = _goal(solver)
    seed = solver.default_joint_position.reshape(1, 1, -1)
    first = solver.solve_single(goal, seed_config=seed)
    solver.destroy()
    assert solver._idxs_goal is None
    second = solver.solve_single(goal, seed_config=seed)
    assert first.success.all() and second.success.all()
    torch.testing.assert_close(first.solution, second.solution)


def test_current_state_contract_rejects_invalid_time_and_seed_values():
    solver = SeedIKSolver(_cfg(velocity_weight=0.1, device_cfg=DeviceCfg("cpu")))
    goal = _goal(solver)
    position = solver.default_joint_position.reshape(1, -1)
    with pytest.raises(ValueError, match="positive finite"):
        solver.solve_single(
            goal,
            current_state=JointState(position, dt=torch.zeros(1), device_cfg=DeviceCfg("cpu")),
            seed_config=position,
        )
    with pytest.raises(ValueError, match="finite"):
        solver.solve_single(goal, seed_config=torch.full((1, 1, solver.dof), float("nan")))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_destroy_recreate_and_velocity_validation_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"), torch.float32)
    solver = SeedIKSolver(_cfg(device_cfg, velocity_weight=0.1))
    goal = _goal(solver)
    position = solver.default_joint_position.reshape(1, -1)
    result = solver.solve_single(
        goal,
        current_state=JointState(position, velocity=torch.zeros_like(position), dt=torch.ones(1, device="mps")),
        seed_config=position,
    )
    solver.destroy()
    repeated = solver.solve_single(goal, seed_config=position)
    assert result.solution.device.type == repeated.solution.device.type == "mps"
    assert result.success.all()
    assert torch.isfinite(repeated.solution).all()
