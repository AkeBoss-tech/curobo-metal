"""Portable lifecycle coverage for PRM's indexed-roadmap debugging hooks."""

from __future__ import annotations

from curobo.types import DeviceCfg

import pytest
import torch

from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg


def _planner(feasible=None) -> PRMGraphPlanner:
    return PRMGraphPlanner(
        PRMGraphPlannerCfg(
            max_nodes=12,
            action_lower_bounds=torch.tensor([-1.0]),
            action_upper_bounds=torch.tensor([1.0]),
            check_feasibility_fn=feasible,
            edge_step=0.05,
            neighbors_per_node=2,
            sampler_seed=3,
            use_cuda_graph_for_rollout=False,
            device_cfg=DeviceCfg("cpu"),
        )
    )


def test_indexed_roadmap_graph_is_weighted_deterministic_and_resettable() -> None:
    planner = _planner()
    planner._append_samples(torch.tensor([[-0.9], [-0.3], [0.3], [0.9]]))

    assert planner._compat_graph_generation != planner._generation
    assert planner.graph_path_finder.graph.number_of_nodes() == 4
    assert planner._compat_graph_generation == planner._generation
    exists, labels = planner._check_paths_exist([0, 0], [3, 2], require_all_paths=True)
    assert exists is True
    assert labels == [True, True]

    paths, lengths = planner._find_path_for_index_pairs([0, 0], [3, 2], return_length=True)
    # Equal-cost alternatives use the path finder's documented
    # lexicographically-first tie rule.
    assert paths == [[0, 1, 2, 3], [0, 1, 2]]
    assert lengths == pytest.approx([1.8, 1.2])
    # The wrapper keeps a stable networkx-shaped view and returns the same
    # indexed result on repeat rather than rebuilding a query-local graph.
    assert planner._find_path_for_index_pairs([0], [3]) == [paths[0]]

    planner.reset_buffer()
    assert planner.n_nodes == 0
    assert planner.graph_path_finder.graph.number_of_nodes() == 0
    assert planner._check_paths_exist([0], [3]) == (False, [False])


def test_repeated_extensions_defer_one_compatibility_rebuild_until_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner()
    refresh_count = 0
    original_refresh = planner._refresh_compat_graph

    def counted_refresh() -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh()

    monkeypatch.setattr(planner, "_refresh_compat_graph", counted_refresh)
    planner._append_samples(torch.tensor([[-0.9], [-0.3]]))
    planner._append_samples(torch.tensor([[0.3], [0.9]]))
    assert refresh_count == 0

    assert planner.graph_path_finder.graph.number_of_nodes() == 4
    assert refresh_count == 1
    assert planner.graph_path_finder.graph.number_of_edges() > 0
    assert refresh_count == 1


def test_auto_reset_discards_dirty_compatibility_graph_without_building_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PRMGraphPlanner(
        PRMGraphPlannerCfg(
            max_nodes=40,
            action_lower_bounds=torch.tensor([-1.0]),
            action_upper_bounds=torch.tensor([1.0]),
            new_nodes_per_iteration=0,
            neighbors_per_node=2,
            use_cuda_graph_for_rollout=False,
            device_cfg=DeviceCfg("cpu"),
        )
    )
    refresh_count = 0
    original_refresh = planner._refresh_compat_graph

    def counted_refresh() -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh()

    monkeypatch.setattr(planner, "_refresh_compat_graph", counted_refresh)
    for _ in range(4):
        planner.extend_roadmap_with_random_samples(8, neighbors_per_node=2)
    assert planner.n_nodes == 32
    assert refresh_count == 0

    result = planner.find_path(
        torch.tensor([[-0.8]]),
        torch.tensor([[0.8]]),
        interpolate_waypoints=False,
    )
    assert result.success.tolist() == [True]
    assert planner.n_nodes == 0
    assert refresh_count == 0
    assert planner.graph_path_finder.graph.number_of_nodes() == 0


def test_indexed_roadmap_rechecks_edge_feasibility_and_keeps_partial_connectivity() -> None:
    def feasible(points: torch.Tensor) -> torch.Tensor:
        return points[:, 0].abs() >= 0.2

    planner = _planner(feasible)
    # Every vertex is valid, but the central obstacle invalidates every
    # candidate bridge between the negative and positive components.
    planner._append_samples(torch.tensor([[-0.9], [-0.4], [0.4], [0.9]]))
    any_path, labels = planner._check_paths_exist([0, 0], [1, 3])
    assert any_path is True
    assert labels == [True, False]
    all_paths, all_labels = planner._check_paths_exist([0, 0], [1, 3], require_all_paths=True)
    assert all_paths is False
    assert all_labels == labels
    paths, lengths = planner._find_path_for_index_pairs([0, 0], [1, 3], return_length=True)
    assert paths[0] == [0, 1]
    assert paths[1] is None
    assert lengths[0] == pytest.approx(0.5)
    assert lengths[1] == float("inf")


def test_indexed_roadmap_validates_pair_lists() -> None:
    planner = _planner()
    with pytest.raises(ValueError, match="Start and Goal idx length"):
        planner._find_path_for_index_pairs([0, 1], [2])
    with pytest.raises(ValueError, match="Start and Goal idx length"):
        planner._check_paths_exist([0], [1, 2])
    with pytest.raises(TypeError, match="require_all_paths"):
        planner._check_paths_exist([], [], require_all_paths=1)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_indexed_roadmap_edge_validation_remains_on_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    planner = _planner()
    planner.config.action_lower_bounds = planner.config.action_lower_bounds.to("mps")
    planner.config.action_upper_bounds = planner.config.action_upper_bounds.to("mps")
    planner._append_samples(torch.tensor([[-0.8], [-0.2], [0.4], [0.8]], device="mps"))
    assert planner._roadmap_samples is not None
    assert planner._roadmap_samples.device.type == "mps"
    assert planner._check_paths_exist([0], [3]) == (True, [True])
