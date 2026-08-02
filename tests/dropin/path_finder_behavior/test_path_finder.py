"""Behavioural coverage for the portable pinned-V2 NetworkX path facade."""

from __future__ import annotations

import random

import pytest
import torch

from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder


def test_staged_nodes_and_edges_materialize_on_query_and_reset_cleanly() -> None:
    graph = NetworkXPathFinder(seed=13)
    graph.add_nodes([0, 4])
    graph.add_edges([(0, 2, 1.25), (2, 4, 2.5)])

    # Pinned behavior buffers additions until a query/update, and the public
    # graph view makes that lifecycle inspectable without NetworkX.
    assert graph.node_list == [0, 4]
    assert len(graph.edge_list) == 2
    assert graph.graph.number_of_nodes() == 3
    assert graph.node_list == []
    assert graph.edge_list == []
    assert graph.get_edges() == [(0, 2, 1.25), (2, 4, 2.5)]
    assert graph.get_shortest_path(0, 4, return_length=True) == ([0, 2, 4], 3.75)

    graph.reset_graph()
    assert graph.graph.number_of_nodes() == 0
    assert graph.get_edges() == []
    assert graph.get_shortest_path(0, 4) is None


def test_equal_cost_routes_use_canonical_lexicographic_path_independent_of_insertion() -> None:
    first = NetworkXPathFinder()
    first.add_edges([(0, 2, 1.0), (2, 4, 1.0), (0, 1, 1.0), (1, 4, 1.0)])
    second = NetworkXPathFinder()
    second.add_edges([(1, 4, 1.0), (0, 1, 1.0), (2, 4, 1.0), (0, 2, 1.0)])

    assert first.get_shortest_path(0, 4, return_length=True) == ([0, 1, 4], 2.0)
    assert second.get_shortest_path(0, 4, return_length=True) == ([0, 1, 4], 2.0)
    assert first.path_exists(0, 4) is True
    assert first.path_exists(0, 99) is False
    assert first.get_shortest_path(0, 99, return_length=True) == (None, float("inf"))


def test_repeated_edge_updates_weight_and_dense_lengths_match_roadmap_slots() -> None:
    graph = NetworkXPathFinder()
    graph.add_nodes([0, 1, 2, 6])
    graph.add_edge(0, 1, 9.0)
    graph.add_edge(1, 0, 2.0)
    graph.add_edge(1, 2, 3.0)
    graph.update_graph()

    # NetworkX Graph semantics overwrite an existing undirected edge rather
    # than adding a parallel edge.  Unreachable node slots stay at -1.0.
    assert graph.get_edges() == [(0, 1, 2.0), (1, 2, 3.0)]
    assert graph.get_path_lengths(2) == [5.0, 3.0, 0.0, -1.0, -1.0, -1.0, -1.0]
    assert graph.get_path_lengths(42) == []


def test_tensor_inputs_are_accepted_and_invalid_edge_values_fail_precisely() -> None:
    graph = NetworkXPathFinder()
    graph.add_nodes(torch.tensor([0, 1, 2], dtype=torch.int64))
    graph.add_edges(torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 1.5]]))
    assert graph.get_shortest_path(torch.tensor(0), torch.tensor(2), return_length=True) == (
        [0, 1, 2],
        2.5,
    )
    with pytest.raises(ValueError, match="shape"):
        graph.add_edges(torch.zeros((3, 2)))
    with pytest.raises(ValueError, match="negative"):
        graph.add_edge(0, 3, -1.0)
    with pytest.raises(ValueError, match="finite"):
        graph.add_edge(0, 3, float("nan"))


def test_reset_seed_restores_python_random_stream() -> None:
    graph = NetworkXPathFinder(seed=121)
    graph.reset_seed()
    first = random.random()
    random.random()
    graph.reset_seed()
    assert random.random() == first


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_edge_tensor_is_consumed_without_mps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    graph = NetworkXPathFinder()
    edges = torch.tensor([[0.0, 1.0, 0.5], [1.0, 3.0, 0.25]], device="mps")
    graph.add_edges(edges)
    assert graph.get_shortest_path(0, 3, return_length=True) == ([0, 1, 3], 0.75)
    # Search is intentionally a Python control-plane stage; its MPS input has
    # been consumed exactly once and does not require an unsupported kernel.
    assert edges.device.type == "mps"
