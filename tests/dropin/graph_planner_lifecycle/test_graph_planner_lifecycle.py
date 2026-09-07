"""Lifecycle checks for the portable V2-shaped PRM graph planner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class _Transition:
    """Minimal eager transition to prove graph PRM owns configured rollouts."""

    def __init__(self, cfg):
        del cfg
        self.action_dim = 2
        self.action_horizon = self.horizon = 1
        self._dt = torch.tensor([0.1])
        self.action_bound_lows = torch.tensor([-1.0, -1.0])
        self.action_bound_highs = torch.tensor([1.0, 1.0])
        self.default_joint_position = torch.zeros(2)

    def update_batch_size(self, batch_size):
        assert batch_size > 0

    def forward(self, start_state, actions, *args, **kwargs):
        del start_state, args, kwargs
        return JointState.from_position(actions)


def _planner(*, feasible=None, rollout=False, scene_collision_cfg=None) -> PRMGraphPlanner:
    device = DeviceCfg("cpu")
    rollout_cfg = None
    if rollout:
        rollout_cfg = RobotRolloutCfg(
            device_cfg=device,
            transition_model_cfg=SimpleNamespace(class_type=_Transition),
        )
    cfg = PRMGraphPlannerCfg(
        max_nodes=16,
        new_nodes_per_iteration=0,
        action_lower_bounds=torch.tensor([-1.0, -1.0]),
        action_upper_bounds=torch.tensor([1.0, 1.0]),
        check_feasibility_fn=feasible,
        rollout_config=rollout_cfg,
        scene_collision_cfg=scene_collision_cfg,
        device_cfg=device,
    )
    return PRMGraphPlanner(cfg)


def test_valid_query_means_valid_endpoints_not_universal_path_success() -> None:
    def free(point: torch.Tensor) -> torch.Tensor:
        # Terminals at +/- 0.8 are free, but the direct edge is blocked.
        return point[:, 0].abs() > 0.2

    planner = _planner(feasible=free)
    result = planner.find_path(
        torch.tensor([[-0.8, 0.0]]), torch.tensor([[0.8, 0.0]]),
        interpolate_waypoints=False,
    )
    assert result.success.tolist() == [False]
    assert result.valid_query is True
    assert result.failure_indices.tolist() == [0]
    assert result.success_indices.numel() == 0


def test_configured_rollouts_and_scene_checker_are_owned_without_cuda_graphs() -> None:
    scene = SceneCollisionCfg(
        scene_model=SceneCfg(cuboid=[Cuboid("wall", [4, 0, 0, 1, 0, 0, 0], [1, 1, 1], device_cfg=DeviceCfg("cpu"))])
    , device_cfg=DeviceCfg("cpu"))
    owned = _planner(rollout=True, scene_collision_cfg=scene)
    assert owned.scene_collision_checker is not None
    assert owned.scene_collision_checker.check_obstacle_exists("wall")
    rollouts = owned.get_all_rollout_instances()
    assert len(rollouts) == 2
    assert all(not item.valid_compute_metrics_from_action_cuda_graph() for item in rollouts)
    assert owned.check_samples_feasibility(torch.tensor([[0.0, 0.0]])).tolist() == [True]


def test_result_rejects_nonfinite_lifecycle_metadata_and_keeps_inf_failure_marker() -> None:
    result = GraphPlannerResult(
        torch.tensor([False]), path_length=torch.tensor([float("inf")]), solve_time=0.0
    )
    assert result.failure_indices.tolist() == [0]
    with pytest.raises(ValueError, match="must not contain NaN"):
        GraphPlannerResult(torch.tensor([False]), path_length=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="finite nonnegative"):
        GraphPlannerResult(torch.tensor([True]), solve_time=-0.1)
    with pytest.raises(ValueError, match="unique"):
        GraphPlannerResult(torch.tensor([True]), joint_names=["joint", "joint"])
