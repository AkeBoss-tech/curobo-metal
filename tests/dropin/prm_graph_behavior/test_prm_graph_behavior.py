"""Behavioral coverage for the portable pinned-V2 PRM façade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.util.trajectory import TrajInterpolationType


def _cfg(*, feasible=None, new_nodes: int = 0, max_nodes: int = 32) -> PRMGraphPlannerCfg:
    return PRMGraphPlannerCfg(
        max_nodes=max_nodes,
        action_lower_bounds=torch.tensor([-1.0, -1.0]),
        action_upper_bounds=torch.tensor([1.0, 1.0]),
        new_nodes_per_iteration=new_nodes,
        neighbors_per_node=8,
        sampler_seed=17,
        sample_rejection_ratio=2,
        use_cuda_graph_for_rollout=False,
        check_feasibility_fn=feasible,
    )


def test_extended_roadmap_is_deterministic_consumed_and_resettable() -> None:
    planner = PRMGraphPlanner(_cfg())
    planner.extend_roadmap_with_random_samples(8)
    first = planner._roadmap_samples.clone()
    assert planner.n_nodes == 8

    result = planner.find_path(
        torch.tensor([[-0.9, -0.1]]), torch.tensor([[0.8, 0.2]]),
        interpolate_waypoints=False,
    )
    assert result.success.tolist() == [True]
    assert result.debug_info["metrics"][0].samples_requested == 8
    assert result.debug_info["metrics"][0].samples_valid == 8

    planner.reset_buffer()
    assert planner.n_nodes == 0
    planner.reset_seed()
    planner.extend_roadmap_with_random_samples(8)
    torch.testing.assert_close(planner._roadmap_samples, first)


def test_extension_obeys_feasibility_bounds_and_capacity() -> None:
    def feasible(points: torch.Tensor) -> torch.Tensor:
        return (points[:, 0] >= 0) & (points[:, 1] <= 0.8)

    planner = PRMGraphPlanner(_cfg(feasible=feasible, max_nodes=16))
    planner.extend_roadmap_with_random_samples(4)
    samples = planner._roadmap_samples
    assert samples.shape[0] <= 4
    assert bool(feasible(samples).all())
    assert bool(((samples >= -1) & (samples <= 1)).all())

    full = PRMGraphPlanner(_cfg(max_nodes=4))
    full.extend_roadmap_with_random_samples(4)
    with pytest.raises(ValueError, match="capacity"):
        full.extend_roadmap_with_random_samples(1)

    planner.reset_buffer()
    planner.extend_roadmap_with_ellipsoidal_samples(
        torch.tensor([-0.2, -0.4]), torch.tensor([0.4, 0.3]), torch.tensor(0.6), 4,
    )
    samples = planner._roadmap_samples
    assert bool(feasible(samples).all())
    assert bool(((samples >= -1) & (samples <= 1)).all())


def test_find_path_preserves_pinned_batch_validity_and_interpolation_contract() -> None:
    def feasible(points: torch.Tensor) -> torch.Tensor:
        # An invalid member invalidates the whole batch in the pinned PRM API.
        return points[:, 0] > -0.75

    planner = PRMGraphPlanner(_cfg(feasible=feasible, new_nodes=4))
    failed = planner.find_path(
        torch.tensor([[-0.9, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.2, 0.0], [0.3, 0.0]]),
    )
    assert failed.success.tolist() == [False, False]
    assert failed.valid_query is False
    assert failed.plan_waypoints == [None, None]

    paths = [
        torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        None,
    ]
    success = torch.tensor([True, False])
    cubic = planner.get_interpolated_trajectory(paths, success, 7, TrajInterpolationType.CUBIC)
    assert cubic.shape == (2, 7, 2)
    torch.testing.assert_close(cubic[0, 0], paths[0][0])
    torch.testing.assert_close(cubic[0, -1], paths[0][-1])
    assert torch.equal(cubic[1], torch.zeros_like(cubic[1]))
    with pytest.raises(ValueError, match="Unsupported interpolation"):
        planner.get_interpolated_trajectory(paths, success, 7, TrajInterpolationType.LINEAR_CUDA)


def test_strict_tensor_ranks_and_warmup_lifecycle() -> None:
    planner = PRMGraphPlanner(_cfg(new_nodes=2))
    with pytest.raises(ValueError, match="2D"):
        planner.find_path(torch.zeros(2), torch.ones(2))
    with pytest.raises(ValueError, match="same shape"):
        planner.find_path(torch.zeros(1, 2), torch.ones(2, 2))
    planner.warmup(num_warmup_iterations=2, max_batch_size=1)
    assert planner.n_nodes == 0
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        planner.reset_cuda_graph()


def test_query_growth_retains_deterministic_roadmap_and_honors_neighbor_policy() -> None:
    def feasible(points: torch.Tensor) -> torch.Tensor:
        # A vertical wall blocks the direct terminal edge but leaves two
        # deterministic routes around it.  It forces the portable PRM through
        # its terminal-first then ellipsoidal-growth lifecycle.
        in_wall = (points[:, 0].abs() < 0.20) & (points[:, 1].abs() < 0.35)
        return ~in_wall

    cfg = _cfg(feasible=feasible, new_nodes=16, max_nodes=96)
    cfg.max_path_finding_iterations = 4
    cfg.exploration_radius = 1.25
    cfg.neighbors_per_node = 4
    cfg.neighbors_per_node_growth_factor = 1.5
    planner = PRMGraphPlanner(cfg)
    result = planner._find_path_impl(
        torch.tensor([[-0.9, 0.0]]), torch.tensor([[0.9, 0.0]])
    )
    assert result.success.tolist() == [True]
    assert planner.n_nodes > 0
    assert result.debug_info["n_nodes"] == planner.n_nodes
    assert result.debug_info["neighbors_per_node"] >= cfg.neighbors_per_node
    assert result.plan_waypoints[0].shape[-1] == 2

    snapshot = planner._roadmap_samples.clone()
    planner.reset_buffer()
    planner.reset_seed()
    repeat = PRMGraphPlanner(cfg)
    repeated = repeat._find_path_impl(
        torch.tensor([[-0.9, 0.0]]), torch.tensor([[0.9, 0.0]])
    )
    assert repeated.success.tolist() == [True]
    torch.testing.assert_close(snapshot, repeat._roadmap_samples)


def test_identical_terminals_are_zero_length_without_roadmap_growth() -> None:
    planner = PRMGraphPlanner(_cfg(new_nodes=8))
    position = torch.tensor([[0.2, -0.4]])
    result = planner._find_path_impl(position, position.clone())
    assert result.success.tolist() == [True]
    assert planner.n_nodes == 0
    assert result.path_length.tolist() == [0.0]
    torch.testing.assert_close(result.plan_waypoints[0], position.expand(2, -1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_prm_roadmap_stays_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = _cfg(new_nodes=4)
    cfg.action_lower_bounds = cfg.action_lower_bounds.to("mps")
    cfg.action_upper_bounds = cfg.action_upper_bounds.to("mps")
    planner = PRMGraphPlanner(cfg)
    planner.extend_roadmap_with_random_samples(4)
    assert planner._roadmap_samples.device.type == "mps"
    result = planner.find_path(
        torch.tensor([[-0.8, 0.0]], device="mps"),
        torch.tensor([[0.8, 0.0]], device="mps"),
    )
    assert result.success.device.type == "mps"
    assert result.interpolated_waypoints.device.type == "mps"
