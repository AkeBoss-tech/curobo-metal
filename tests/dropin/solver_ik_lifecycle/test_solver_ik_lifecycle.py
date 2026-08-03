"""Portable lifecycle coverage for the cuRoboV2 IK facade.

These tests exercise the public SolverCore-shaped delegation as well as the
real Torch IK solve path.  They intentionally make no CUDA graph or Warp ABI
claim: MPS uses the same eager tensor lifecycle as CPU.
"""

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solver(device_cfg=DeviceCfg(), **kwargs):
    kwargs.setdefault("num_seeds", 2)
    return IKSolver(IKSolverCfg.create(
        "franka.yml", device_cfg=device_cfg, use_cuda_graph=False, **kwargs
    ))


def _exact_goal(solver):
    state = solver.compute_kinematics(solver.default_joint_state)
    return GoalToolPose(
        solver.tool_frames, state.tool_poses.position.unsqueeze(3),
        state.tool_poses.quaternion.unsqueeze(3),
    )


def test_ik_delegates_core_goal_seed_tracking_and_inertial_lifecycle():
    solver = _solver()
    assert solver.kinematics is solver.core.kinematics
    assert solver.goal_registry_manager is solver.core.goal_registry_manager
    assert solver.seed_manager is solver.core.seed_manager
    assert solver.problem_batch_size == 2

    seeds = solver.prepare_action_seeds(
        1, 2, seed_config=solver.default_joint_position.unsqueeze(0)
    )
    assert seeds.shape == (2, 1, solver.action_dim)
    solver.enable_tool_pose_tracking()
    assert bool(solver.core._tool_pose_criteria[solver.tool_frames[0]].terminal_pose_axes_weight_factor.any())
    solver.disable_tool_pose_tracking()
    assert not bool(solver.core._tool_pose_criteria[solver.tool_frames[0]].terminal_pose_axes_weight_factor.any())

    link = solver.core.kinematics._model.link_names[0]
    solver.update_link_inertial(link, mass=1.75)
    assert solver.core.kinematics._model.mass[0].item() == pytest.approx(1.75)
    # Raw CUDA graph controls remain an explicit device boundary.
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        solver.reset_cuda_graph()


def test_solve_impl_validates_external_state_and_unique_solution_is_stable():
    solver = _solver()
    goal = _exact_goal(solver)
    state = SolveState(
        SolveMode.SINGLE, 1, 1, num_ik_seeds=solver.config.num_seeds,
        tool_frames=list(solver.tool_frames),
    )
    result = solver._solve_impl(
        state, goal, solver.config.num_seeds,
        current_state=solver.default_joint_state, run_optimizer=False,
    )
    assert result.success.tolist() == [[True]]
    assert solver.solve_state is not None
    unique = solver.get_unique_solution()
    assert unique.shape == (1, solver.action_dim)
    torch.testing.assert_close(unique[0], result.solution[0, 0])

    invalid = SolveState(SolveMode.SINGLE, 2, 1, num_ik_seeds=2, tool_frames=list(solver.tool_frames))
    with pytest.raises(ValueError, match="batch_size"):
        solver._solve_impl(invalid, goal, 2, run_optimizer=False)
    with pytest.raises(ValueError, match="config.num_seeds"):
        solver._solve_impl(state, goal, 3, run_optimizer=False)


def test_factory_keeps_structured_regularization_overrides_in_optimizer_records():
    cfg = IKSolverCfg.create(
        "franka.yml",
        optimizer_configs=[{"rollout": {"cost_cfg": {"cspace_cfg": {}}}}],
        velocity_regularization_weight=0.2,
        acceleration_regularization_weight=0.3,
    )
    assert cfg.optimizer_configs[0]["rollout"]["cost_cfg"]["cspace_cfg"][
        "squared_l2_regularization_weight"
    ] == [0.2, 0.3]
    with pytest.raises(ValueError, match="squared_l2"):
        IKSolverCfg.create(
            "franka.yml",
            optimizer_configs=[{"rollout": {"cost_cfg": {"cspace_cfg": {
                "squared_l2_regularization_weight": [1.0]
            }}}}],
            velocity_regularization_weight=0.2,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_ik_core_lifecycle_and_internal_solve_stay_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    solver = _solver(DeviceCfg(torch.device("mps"), torch.float32))
    goal = _exact_goal(solver)
    state = SolveState(
        SolveMode.SINGLE, 1, 1, num_ik_seeds=solver.config.num_seeds,
        tool_frames=list(solver.tool_frames),
    )
    result = solver._solve_impl(state, goal, solver.config.num_seeds, run_optimizer=False)
    assert result.solution.device.type == "mps"
    assert solver.core.goal_buffer.link_goal_poses.device.type == "mps"
