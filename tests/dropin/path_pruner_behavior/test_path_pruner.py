"""Behavioral tests for portable pinned-V2 PRM shortcut pruning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.graph_planner.search.path_pruner import PathPruner
from curobo._src.types.device_cfg import DeviceCfg


def _pruner(device: str = "cpu") -> tuple[PathPruner, list[tuple[torch.Tensor, torch.Tensor]]]:
    cfg = SimpleNamespace(device_cfg=DeviceCfg(device=torch.device(device)))
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    pruner = PathPruner(cfg)
    nodes = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 3.0], [2.0, 0.0, 4.0]],
        **cfg.device_cfg.as_torch_dict(),
    )

    def steer(start: torch.Tensor, goal: torch.Tensor, *, add_exact_node: bool) -> None:
        assert add_exact_node is False
        calls.append((start.clone(), goal.clone()))

    def find(starts, goals, *, return_length: bool):
        assert return_length is True
        return [[int(start), int(goal)] for start, goal in zip(starts, goals)], [1.0] * len(starts)

    pruner.set_dependencies(2, torch.ones(2, **cfg.device_cfg.as_torch_dict()), nodes, steer, find)
    return pruner, calls


def test_shortcut_candidates_are_complete_ordered_and_device_resident() -> None:
    pruner, calls = _pruner()
    paths, lengths = pruner.prune_path_with_shortcuts([[0, 2, 4], [1, 3]], [0, 1], [4, 3])
    assert paths == [[0, 4], [1, 3]]
    assert lengths == [1.0, 1.0]
    assert len(calls) == 1
    start, goal = calls[0]
    assert start.shape == goal.shape == (9, 3)
    assert start.device.type == goal.device.type == "cpu"
    assert list(zip(start[:, -1].tolist(), goal[:, -1].tolist())) == [
        (0.0, 0.0), (0.0, 2.0), (0.0, 4.0), (2.0, 2.0), (2.0, 4.0), (4.0, 4.0),
        (1.0, 1.0), (1.0, 3.0), (3.0, 3.0),
    ]


def test_shortcut_registration_reduces_real_weighted_graph_path() -> None:
    cfg = SimpleNamespace(device_cfg=DeviceCfg())
    finder = NetworkXPathFinder(seed=3)
    finder.add_nodes([0, 1, 2])
    finder.add_edges([(0, 1, 1.1), (1, 2, 1.1)])
    nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    pruner = PathPruner(cfg)

    def steer(start: torch.Tensor, goal: torch.Tensor, *, add_exact_node: bool) -> None:
        assert not add_exact_node
        delta = torch.linalg.vector_norm(start[:, :2] - goal[:, :2], dim=-1)
        finder.add_edges(
            [
                (int(a.item()), int(b.item()), float(distance.item()))
                for a, b, distance in zip(start[:, -1], goal[:, -1], delta)
            ]
        )

    def find(starts, goals, *, return_length: bool):
        return [finder.get_shortest_path(s, g, return_length=return_length)[0] for s, g in zip(starts, goals)], [
            finder.get_shortest_path(s, g, return_length=return_length)[1] for s, g in zip(starts, goals)
        ]

    pruner.set_dependencies(2, torch.ones(2), nodes, steer, find)
    paths, lengths = pruner.prune_path_with_shortcuts([[0, 1, 2]], [0], [2])
    assert paths == [[0, 2]]
    assert lengths == [2.0]


def test_validation_empty_paths_and_dependency_lifecycle() -> None:
    pruner, calls = _pruner()
    empty_paths, empty_lengths = pruner.prune_path_with_shortcuts([], [], [])
    assert empty_paths == empty_lengths == []
    assert calls == []
    with pytest.raises(ValueError, match="one entry per path"):
        pruner.prune_path_with_shortcuts([[0]], [], [0])
    with pytest.raises(ValueError, match="within the supplied node buffer"):
        pruner._prepare_edges_for_shortcuts([[9]])
    with pytest.raises(TypeError, match="integer"):
        pruner._prepare_edges_for_shortcuts([[0.0]])

    bare = PathPruner(SimpleNamespace(device_cfg=DeviceCfg()))
    with pytest.raises(RuntimeError, match="set_dependencies"):
        bare.prune_path_with_shortcuts([], [], [])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_shortcut_batches_stay_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    pruner, calls = _pruner("mps")
    paths, lengths = pruner.prune_path_with_shortcuts([[0, 2, 4]], [0], [4])
    assert paths == [[0, 4]]
    assert lengths == [1.0]
    assert calls[0][0].device.type == calls[0][1].device.type == "mps"
