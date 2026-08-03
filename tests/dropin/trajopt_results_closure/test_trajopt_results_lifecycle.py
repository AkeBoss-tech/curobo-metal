"""Regression coverage for portable TrajOpt result lifecycle boundaries."""

import pytest
import torch

from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState


def _reduced_result(device: str = "cpu") -> TrajOptSolverResult:
    """Model a solver that returned two trajectories from four attempted seeds."""
    batch, all_seeds, returned, horizon, dof = 2, 4, 2, 4, 2
    attempted = torch.arange(
        batch * all_seeds * horizon * dof, device=device, dtype=torch.float32
    ).reshape(batch, all_seeds, horizon, dof)
    # Stable rank for costs [3, 1, 4, 2] is original seed ids [1, 3, 0, 2].
    selected = attempted[:, torch.tensor([1, 3], device=device)]
    state = JointState(
        selected.clone(), velocity=selected.add(10), acceleration=selected.add(20),
        jerk=selected.add(30), joint_names=["j0", "j1"],
        dt=torch.full((batch, returned), 0.1, device=device),
    )
    return TrajOptSolverResult(
        success=torch.tensor([[True, False], [True, True]], device=device),
        solution=selected.clone(), js_solution=state,
        interpolated_trajectory=state.clone(),
        interpolated_last_tstep=torch.full((batch, returned), horizon, dtype=torch.long, device=device),
        feasible=torch.tensor([[True, True], [True, True]], device=device),
        seed_cost=torch.tensor([[3.0, 1.0, 4.0, 2.0], [3.0, 1.0, 4.0, 2.0]], device=device),
        total_cost_reshaped=torch.tensor(
            [[3.0, 1.0, 4.0, 2.0], [3.0, 1.0, 4.0, 2.0]], device=device
        ),
        optimized_seeds=attempted,
        batch_size=batch, num_seeds=all_seeds,
        debug_info={"per_seed_cost": torch.tensor([[30.0, 10.0, 40.0, 20.0]] * batch, device=device)},
    )


def test_process_normalizes_reduced_return_seed_buffers_and_debug_payloads():
    result = _reduced_result()
    result.process_metrics_and_rank_seeds()

    assert result.num_seeds == 2
    assert result.seed_cost.tolist() == [[1.0, 2.0], [1.0, 2.0]]
    # Original optimizer ids remain visible for replay/debug correlation.
    assert result.seed_rank.tolist() == [[1, 3], [1, 3]]
    assert result.optimized_seeds.shape[:2] == (2, 2)
    assert result.debug_info["per_seed_cost"].tolist() == [[10.0, 20.0], [10.0, 20.0]]
    result.validate()

    best = result.best_seed()
    assert best.solution.shape[:2] == (2, 1)
    assert best.seed_cost.tolist() == [[1.0], [1.0]]
    assert best.seed_rank.tolist() == [[1], [1]]
    assert best.debug_info["per_seed_cost"].shape == (2, 1)


def test_select_batch_detach_and_per_timestep_motion_duration_are_value_safe():
    result = _reduced_result()
    result.process_metrics_and_rank_seeds()
    # A real schedule has one interval entry per materialised horizon entry.
    result.js_solution.dt = torch.tensor(
        [[[0.1, 0.2, 0.3, 9.0], [0.2, 0.3, 0.4, 9.0]],
         [[0.3, 0.4, 0.5, 9.0], [0.4, 0.5, 0.6, 9.0]]]
    )
    assert torch.allclose(result.motion_time(), torch.tensor([[0.6, 0.9], [1.2, 1.5]]))

    selected = result.select_batch([1])
    assert selected.solution.shape[:2] == (1, 2)
    assert selected.debug_info["per_seed_cost"].shape == (1, 2)
    detached = selected.detach()
    selected.solution[0, 0, 0, 0] = -9.0
    assert detached.solution[0, 0, 0, 0] != -9.0
    assert not detached.solution.requires_grad


def test_zero_interpolated_last_tstep_preserves_full_trajectory_and_copy_supports_batch_dt():
    target, source = _reduced_result(), _reduced_result()
    target.interpolated_last_tstep.zero_()
    assert target.get_interpolated_plan().position.shape[-2] == 4

    # dt can be one value per batch even when trajectory positions include a
    # seed axis.  Merging one successful seed should copy that batch timing.
    target.js_solution.dt = torch.tensor([0.1, 0.1])
    source.js_solution.dt = torch.tensor([0.7, 0.8])
    target.process_metrics_and_rank_seeds()
    source.process_metrics_and_rank_seeds()
    source.success = torch.tensor([[False, True], [False, False]])
    target.copy_successful_solutions(source)
    assert torch.allclose(target.js_solution.dt, torch.tensor([0.7, 0.1]))


def test_production_solver_result_is_normalized_after_return_seed_narrowing():
    """The normalizer covers the real solve path, not only a synthetic value."""
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.solver.solver_trajopt import TrajOptSolver
    from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util.trajectory import TrajInterpolationType

    kinematics = KinematicsCfg.from_robot_yaml_file("franka.yml")
    robot = RobotCfg(kinematics.kinematics_config.robot_cfg, device_cfg=DeviceCfg())
    solver = TrajOptSolver(TrajOptSolverCfg(
        core_cfg=[], robot_config=robot, num_seeds=2, action_horizon=5,
        max_iterations=1, interpolation_type=TrajInterpolationType.LINEAR_CUDA,
        optimizer_name="adam", use_cuda_graph_value=False,
    ))
    start = solver.default_joint_state.unsqueeze(0)
    goal = JointState.from_position(start.position + 0.01, solver.joint_names)
    result = solver.solve_cspace(goal, start, num_seeds=2, return_seeds=1, initial_iters=1)
    assert result.solution.shape[:2] == (1, 1)
    assert result.seed_cost.shape == (1, 1)
    assert result.seed_rank.shape == (1, 1)
    assert result.optimized_seeds.shape[:2] == (1, 1)
    result.validate()
    assert result.get_topk_seeds(1) is result


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_reduced_result_lifecycle_uses_mps_without_cpu_fallback():
    result = _reduced_result("mps")
    result.process_metrics_and_rank_seeds()
    best = result.best_seed().select_batch([1])
    assert best.solution.device.type == "mps"
    assert best.debug_info["per_seed_cost"].device.type == "mps"
