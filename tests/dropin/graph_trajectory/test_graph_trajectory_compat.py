import inspect

import pytest
import torch

from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_sampling_strategy import NodeSamplingStrategy
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import (
    TrajInterpolationType,
    calculate_traj_steps,
    get_batch_interpolated_trajectory,
)
from curobo._src.util.trajectory_execution_manager import TrajectoryExecutionManager
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator
from curobo.trajectory_optimizer import (
    TrajectoryOptimizer, TrajectoryOptimizerCfg, TrajectoryOptimizerResult,
)


def test_pinned_public_names_and_signatures():
    assert TrajectoryOptimizer is TrajOptSolver
    assert TrajectoryOptimizerCfg is TrajOptSolverCfg
    assert TrajectoryOptimizerResult.__name__ == "TrajOptSolverResult"
    assert list(TrajInterpolationType) == [
        TrajInterpolationType.LINEAR, TrajInterpolationType.CUBIC,
        TrajInterpolationType.QUARTIC, TrajInterpolationType.QUINTIC,
        TrajInterpolationType.LINEAR_CUDA,
        TrajInterpolationType.BSPLINE_KNOTS_CUDA,
    ]
    signature = inspect.signature(PRMGraphPlanner.find_path)
    assert signature.parameters["interpolation_steps"].default == 100
    assert signature.parameters["validate_interpolated_trajectory"].default is True


def test_seed_interpolation_and_retiming_are_batched_and_deterministic():
    cfg = DeviceCfg()
    generator = TrajectorySeedGenerator(5, 2, cfg)
    start = torch.tensor([[0.0, -1.0], [1.0, 2.0]])
    goal = torch.tensor([[[1.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [-1.0, 1.0]]])
    seeds = generator.generate_interpolated_seeds(start, goal, 2)
    assert seeds.shape == (2, 2, 5, 2)
    torch.testing.assert_close(seeds[:, :, 0], start[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(seeds[:, :, -1], goal)

    raw = JointState.from_position(seeds[:, 0], ["a", "b"])
    raw.dt = torch.tensor([0.1, 0.2])
    steps, maximum = calculate_traj_steps(raw.dt, torch.tensor(0.05), 5)
    assert steps.tolist() == [9, 17]
    output, actual_steps = get_batch_interpolated_trajectory(
        raw, torch.tensor(0.05), TrajInterpolationType.LINEAR_CUDA, device_cfg=cfg,
    )
    assert output.position.shape == (2, int(maximum), 2)
    assert torch.equal(actual_steps, steps)


def test_graph_primitives_and_prm_lifecycle():
    device_cfg = DeviceCfg()
    low, high = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    cfg = PRMGraphPlannerCfg(
        action_lower_bounds=low, action_upper_bounds=high,
        new_nodes_per_iteration=32, neighbors_per_node=8,
        sampler_seed=7, use_cuda_graph_for_rollout=False,
    )
    sampler = NodeSamplingStrategy(
        cfg, low, high, torch.ones(2), 2,
        lambda value: torch.ones(value.shape[0], dtype=torch.bool), device_cfg,
    )
    first = sampler.generate_action_samples(4)
    sampler.reset_seed()
    torch.testing.assert_close(first, sampler.generate_action_samples(4))
    distance = DistanceNeighborCalculator(2, torch.ones(2), device_cfg)
    torch.testing.assert_close(
        distance.calculate_weighted_distance(torch.zeros(2), torch.tensor([[3.0, 4.0]])),
        torch.tensor([5.0]),
    )
    graph = NetworkXPathFinder()
    graph.add_edges([(0, 1, 2.0), (1, 2, 1.0), (0, 2, 9.0)])
    assert graph.get_shortest_path(0, 2, return_length=True) == ([0, 1, 2], 3.0)

    planner = PRMGraphPlanner(cfg)
    result = planner.find_path(
        torch.tensor([[-0.8, 0.0], [-0.5, 0.2]]),
        torch.tensor([[0.8, 0.0], [0.5, -0.2]]),
        interpolation_steps=25,
    )
    assert result.success.tolist() == [True, True]
    assert result.interpolated_waypoints.shape == (2, 25, 2)
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        planner.reset_cuda_graph()


def test_execution_manager_consumes_commands():
    state = JointState.from_position(torch.arange(12.0).reshape(1, 6, 2))
    manager = TrajectoryExecutionManager(6, command_start_idx=1, command_end_idx=4)
    manager.update_state_action_buffers(state, state.position.clone())
    assert manager.has_valid_next_command()
    torch.testing.assert_close(manager.get_next_command().position, state.position[:, 1])
    assert manager.get_command_sequence().shape == (1, 3, 2)


def test_trajectory_optimizer_cspace_routes_to_production_ops():
    kinematics = KinematicsCfg.from_robot_yaml_file("franka.yml")
    robot = RobotCfg(kinematics.kinematics_config.robot_cfg, device_cfg=DeviceCfg())
    cfg = TrajOptSolverCfg(
        core_cfg=[], robot_config=robot, num_seeds=2,
        interpolation_type=TrajInterpolationType.LINEAR_CUDA,
        action_horizon=6, max_iterations=2, optimizer_name="adam",
        use_cuda_graph_value=False,
    )
    solver = TrajOptSolver(cfg)
    start = solver.default_joint_state.unsqueeze(0)
    goal = JointState.from_position(start.position + 0.01, solver.joint_names)
    result = solver.solve_cspace(goal, start, num_seeds=2, dt=0.05, initial_iters=2)
    assert result.solution.shape == (1, 1, 6, 7)
    torch.testing.assert_close(result.solution[:, :, 0], start.position[:, None])
    torch.testing.assert_close(result.solution[:, :, -1], goal.position[:, None])
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        solver.reset_cuda_graph()
