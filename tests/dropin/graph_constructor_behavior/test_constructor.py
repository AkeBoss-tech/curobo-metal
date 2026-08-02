"""Behavioural tests for the portable pinned-V2 GraphConstructor lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.graph_planner.graph.constructor import GraphConstructor
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import GraphNodeManager
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class _ActionConnector:
    """Action-only connector proving constructor index normalization."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def steer_until_infeasible(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        self.calls.append((start.clone(), goal.clone()))
        return goal[:, :-1]


def _constructor(device: str = "cpu", *, default: bool = True):
    device_cfg = DeviceCfg(device=torch.device(device))
    lower = torch.tensor([-1.0, -1.0], **device_cfg.as_torch_dict())
    cfg = SimpleNamespace(
        action_lower_bounds=lower,
        max_nodes=32,
        steer_buffer_size=32,
        cspace_similarity_threshold=1.0e-4,
        use_default_position_heuristic=default,
        connect_terminal_nodes_with_nearest=True,
        neighbors_per_node=2,
        device_cfg=device_cfg,
    )
    distance = DistanceNeighborCalculator(2, torch.ones(2, **device_cfg.as_torch_dict()), device_cfg)
    manager = GraphNodeManager(cfg, distance, NetworkXPathFinder(seed=7), device_cfg=device_cfg)
    connector = _ActionConnector()
    constructor = GraphConstructor(
        cfg, connector, distance, manager, 2,
        lambda rows: torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device),
        device_cfg,
    )
    return constructor, connector


def test_terminal_lifecycle_caches_default_and_registers_bidirectional_edges() -> None:
    constructor, connector = _constructor()
    default = JointState.from_position(torch.tensor([0.0, 0.0]))
    starts, goals = constructor.initialize_terminal_graph_connections(
        torch.tensor([[-0.8, 0.1], [-0.5, -0.2]]),
        torch.tensor([[0.7, 0.2], [0.4, -0.7]]),
        default,
    )
    assert starts.shape == goals.shape == (2, 3)
    assert constructor._default_joint_position_feasible is True
    assert constructor._default_node_in_roadmap is not None
    assert constructor.node_manager.n_nodes == 5
    assert connector.calls
    graph = constructor.node_manager.get_connected_graph()
    assert graph is not None
    # Direct terminal pairs, default bridges, and nearest-node expansion all
    # flow through real graph registration rather than only returning rows.
    assert graph.connectivity.shape[0] >= 6
    assert graph.nodes.device.type == "cpu"
    cached, feasible = constructor.initialize_default_node(default)
    assert feasible is True
    assert cached is constructor._default_node_in_roadmap


def test_connection_batch_uses_nearest_rows_and_preserves_candidate_indices() -> None:
    constructor, connector = _constructor(default=False)
    initial = constructor.node_manager.add_initial_exact_nodes_to_roadmap(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    constructor.connect_nodes(torch.tensor([[0.9, 0.1], [0.1, 0.9]]), neighbors_per_node=2)
    assert len(connector.calls) == 1
    received_starts, received_goals = connector.calls[0]
    assert received_starts.shape == received_goals.shape == (4, 3)
    assert set(received_starts[:, -1].tolist()).issubset(set(initial[:, -1].tolist()))
    # Action-only connector returns two columns; GraphConstructor restores the
    # candidate row's graph index before GraphNodeManager receives it.
    graph = constructor.node_manager.get_connected_graph()
    assert graph is not None
    assert graph.connectivity.shape[0] == 4
    assert constructor.node_manager.n_nodes == 5


def test_validation_empty_and_reset_preserve_roadmap_but_clear_default_cache() -> None:
    constructor, _ = _constructor()
    default = JointState.from_position(torch.tensor([0.0, 0.0]))
    constructor.initialize_terminal_graph_connections(
        torch.tensor([[-0.4, 0.1]]), torch.tensor([[0.5, -0.1]]), default
    )
    count = constructor.node_manager.n_nodes
    empty = torch.empty((0, 2))
    constructor.connect_nodes(empty)
    assert constructor.node_manager.n_nodes == count
    constructor.reset()
    assert constructor.node_manager.n_nodes == count
    assert constructor._default_joint_position_feasible is None
    with pytest.raises(ValueError, match="same batch size"):
        constructor.steer_and_register_edges(torch.zeros(1, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positive integer"):
        constructor.connect_nodes(torch.zeros(1, 2), neighbors_per_node=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_constructor_terminals_stay_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    constructor, connector = _constructor("mps")
    default = JointState.from_position(torch.tensor([0.0, 0.0], device="mps"))
    starts, goals = constructor.initialize_terminal_graph_connections(
        torch.tensor([[-0.6, 0.1]], device="mps"),
        torch.tensor([[0.6, -0.1]], device="mps"),
        default,
    )
    assert starts.device.type == goals.device.type == "mps"
    assert constructor.node_manager.valid_node_buffer.device.type == "mps"
    assert connector.calls[0][0].device.type == "mps"
