"""Smoke tests for the public graph-planning example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _example_module():
    path = Path(__file__).parents[2] / "examples" / "graph_planning.py"
    spec = importlib.util.spec_from_file_location("graph_planning_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_planning_example_finds_detour_and_builds_seed() -> None:
    path, seeds = _example_module().plan_example(torch.device("cpu"))

    assert path.shape[1] == 2
    assert path.shape[0] > 2
    assert seeds.shape == (1, 1, 32, 2)
    torch.testing.assert_close(seeds[0, 0, 0], path.new_tensor([-1.5, 0.0]))
    torch.testing.assert_close(seeds[0, 0, -1], path.new_tensor([1.5, 0.0]))
