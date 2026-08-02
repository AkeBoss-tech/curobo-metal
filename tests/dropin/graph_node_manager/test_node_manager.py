"""Focused behavioural tests for the portable pinned-V2 graph node manager."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import (
    GraphNodeManager,
    jit_add_all_nodes_to_buffer,
)
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.types.device_cfg import DeviceCfg


def _manager(device: str = "cpu") -> GraphNodeManager:
    cfg = DeviceCfg(device=torch.device(device))
    lower = torch.tensor([-1.0, -1.0], **cfg.as_torch_dict())
    graph_cfg = SimpleNamespace(
        action_lower_bounds=lower,
        max_nodes=8,
        steer_buffer_size=16,
        cspace_similarity_threshold=1.0e-4,
        device_cfg=cfg,
    )
    distance = DistanceNeighborCalculator(2, torch.ones(2, **cfg.as_torch_dict()), cfg)
    return GraphNodeManager(graph_cfg, distance, NetworkXPathFinder(seed=8), device_cfg=cfg)


def test_initial_nodes_preserve_indices_and_exact_duplicate_mapping() -> None:
    manager = _manager()
    rows = torch.tensor([[0.0, 0.0], [0.5, 0.2], [0.0, 0.0]])
    node_set = manager.add_initial_exact_nodes_to_roadmap(rows)
    assert manager.n_nodes == 2
    assert node_set.shape == (3, 3)
    assert node_set[:, -1].tolist() == [0.0, 1.0, 0.0]
    assert manager.valid_node_buffer[:, -1].tolist() == [0.0, 1.0]


def test_roadmap_similarity_and_batch_duplicates_map_to_first_vertex() -> None:
    manager = _manager()
    manager.add_initial_exact_nodes_to_roadmap(torch.tensor([[0.0, 0.0]]))
    returned = manager.add_nodes_to_roadmap(
        torch.tensor([[0.00005, 0.0], [0.4, -0.3], [0.4, -0.3]])
    )
    assert manager.n_nodes == 2
    assert returned[:, -1].tolist() == [0.0, 1.0, 1.0]
    torch.testing.assert_close(manager.valid_node_buffer[1, :2], torch.tensor([0.4, -0.3]))


def test_connection_registration_materializes_weighted_connected_graph() -> None:
    manager = _manager()
    starts = manager.add_initial_exact_nodes_to_roadmap(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    candidates = torch.tensor([[0.0, 1.0, -7.0]])
    assert manager.register_nodes_and_connections(candidates, starts[:1]) is True

    graph = manager.get_connected_graph()
    assert graph is not None
    assert graph.nodes.shape == (3, 2)
    assert graph.edges.shape == (1, 2, 2)
    assert graph.connectivity.shape == (1, 3)
    assert graph.connectivity[0, :2].tolist() == [0.0, 2.0]
    assert graph.connectivity[0, 2].item() == pytest.approx(1.0)
    graph.set_shortest_path_lengths(torch.tensor([2.0, 1.0, 0.0]))
    assert graph.get_node_distance().shape == (3, 3)


def test_reset_path_queries_and_helper_validation() -> None:
    manager = _manager()
    manager.add_initial_exact_nodes_to_roadmap(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    paths = manager.get_nodes_in_path([[1, 0], None])
    assert paths[1] is None
    torch.testing.assert_close(paths[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="valid"):
        manager.get_nodes_in_path([[2]])
    manager.reset_buffer()
    assert manager.n_nodes == 0
    assert manager.get_connected_graph() is None

    buffer = torch.zeros((3, 3))
    updated, rows, used = jit_add_all_nodes_to_buffer(torch.tensor([[3.0, 4.0]]), buffer, 1, 2)
    assert used == 2
    assert rows.tolist() == [[3.0, 4.0, 1.0]]
    assert updated[1].tolist() == [3.0, 4.0, 1.0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_node_lifecycle_stays_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    manager = _manager("mps")
    initial = manager.add_initial_exact_nodes_to_roadmap(
        torch.tensor([[0.0, 0.0]], device="mps")
    )
    returned = manager.add_nodes_to_roadmap(torch.tensor([[0.5, 0.1]], device="mps"))
    assert initial.device.type == "mps"
    assert returned.device.type == "mps"
    assert manager.valid_node_buffer.device.type == "mps"
