"""Focused public planning-stack compatibility checks.

These cover the portable API shape rather than CUDA graph internals, which are
intentionally exposed as explicit unsupported boundaries on CPU/MPS.
"""

import pytest
import torch

from curobo._src.graph_planner import (
    DistanceNeighborCalculator,
    NodeSamplingStrategy,
)
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver import IKSolver, TrajOptSolver


def test_planning_namespace_reexports_and_tensor_helpers():
    values = torch.tensor([[0.0], [0.0], [2.0]])
    unique, indices = DistanceNeighborCalculator.jit_get_unique_nodes_zero_distance(
        values, torch.ones(1), 1
    )
    assert unique.tolist() == [[0.0], [2.0]]
    assert indices.tolist() == [0, 2]
    transformed = NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_svd(
        torch.zeros(2), torch.ones(2), torch.ones(2), torch.tensor(0.5), 2,
        torch.eye(2), torch.tensor([[1.0, -1.0]]), torch.full((2,), -10.0), torch.full((2,), 10.0),
    )
    torch.testing.assert_close(transformed, torch.tensor([[1.0, 0.0]]))


def test_rollout_shape_surface_and_cuda_boundary():
    rollout = RobotRollout()
    assert rollout.action_bounds == (None, None)
    assert rollout.valid_compute_metrics_from_action_cuda_graph() is False
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        rollout.reset_cuda_graph()

    metrics = RolloutMetrics(
        actions=torch.ones(2, 2, 3), feasible=torch.ones(2, 2, dtype=torch.bool)
    )
    selected = metrics.get_only_batch_seed_indices(1, 0)
    assert selected.actions.shape == (3,)
    assert bool(selected.feasible)


def test_solver_namespace_is_publicly_routable():
    assert IKSolver.__name__ == "IKSolver"
    assert TrajOptSolver.__name__ == "TrajOptSolver"
