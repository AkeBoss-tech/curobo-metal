"""Behavioral coverage for the portable pinned TrajOpt facade."""

import json

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType


def _solver(*, horizon=6, seeds=3, interpolation_buffer_size=1000):
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml")
    robot = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=DeviceCfg())
    cfg = TrajOptSolverCfg(
        core_cfg=[], robot_config=robot, num_seeds=seeds, action_horizon=horizon,
        max_iterations=2, optimizer_name="adam", use_cuda_graph_value=False,
        interpolation_type=TrajInterpolationType.LINEAR_CUDA,
        interpolation_buffer_size=interpolation_buffer_size,
    )
    return TrajOptSolver(cfg)


def test_cspace_ranks_seed_results_and_returns_dense_state():
    solver = _solver()
    start = solver.default_joint_state.unsqueeze(0)
    goal = JointState.from_position(start.position + 0.015, solver.joint_names)
    seeds = solver.prepare_action_seeds(1, 3, current_state=start)
    # Make the interior of the first seed deliberately less smooth than the
    # other deterministic seeds; returned plans must be ranked by real cost.
    seeds[0, 0, 2, 0] += 1.0
    result = solver.solve_cspace(goal, start, seed_traj=seeds, return_seeds=2, dt=0.05)
    assert result.solution.shape == (1, 2, 6, 7)
    assert result.seed_rank.shape == (1, 3)
    torch.testing.assert_close(result.seed_rank, result.seed_cost.argsort(dim=-1))
    assert result.interpolated_trajectory.position.shape[:2] == (1, 2)
    assert result.interpolated_last_tstep.shape == (1, 2)
    torch.testing.assert_close(result.solution[:, :, 0], start.position[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(result.solution[:, :, -1], goal.position[:, None].expand(-1, 2, -1))


def test_seed_state_reordering_sampling_and_reset_are_deterministic(tmp_path):
    solver = _solver()
    values = torch.arange(7.0).reshape(1, 7)
    reverse_names = list(reversed(solver.joint_names))
    full = JointState.from_position(values.flip(-1), reverse_names)
    active = solver.get_active_js(full)
    torch.testing.assert_close(active.position, values)
    torch.testing.assert_close(solver.get_full_js(active).position, values)

    first = solver.sample_configs(4)
    solver.reset_seed()
    torch.testing.assert_close(first, solver.sample_configs(4))
    assert torch.all(first >= solver._lower) and torch.all(first <= solver._upper)

    path = tmp_path / "trajopt-debug.json"
    assert solver.debug_dump(path)["cuda_graph"] is False
    assert json.loads(path.read_text())["backend"] == "portable"


def test_dt_validation_retiming_and_interpolation_capacity():
    solver = _solver(horizon=6)
    position = torch.stack((torch.zeros(7), torch.ones(7) * 0.2), dim=0)[None]
    state = JointState.from_position(position, solver.joint_names).finite_difference(0.05)
    dt = solver.compute_trajectory_dt(state)
    assert dt.shape == (1,)
    assert solver.config.minimum_trajectory_dt <= dt.item() <= solver.config.maximum_trajectory_dt
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        solver._resolve_dt(0.0, 1, 1, position)
    with pytest.raises(NotImplementedError, match="per-seed trajectory dt"):
        solver._resolve_dt(torch.tensor([[0.03, 0.04]]), 1, 2, position)

    too_small = _solver(horizon=6, interpolation_buffer_size=2)
    with pytest.raises(ValueError, match="interpolation_buffer_size=2"):
        too_small.get_interpolated_trajectory(state)


def test_config_rejects_invalid_portable_capacities():
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml")
    robot = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=DeviceCfg())
    with pytest.raises(ValueError, match="minimum_trajectory_dt"):
        TrajOptSolverCfg([], robot, minimum_trajectory_dt=0.3, maximum_trajectory_dt=0.1)
    with pytest.raises(ValueError, match="optimizer_name"):
        TrajOptSolverCfg([], robot, optimizer_name="warp")
