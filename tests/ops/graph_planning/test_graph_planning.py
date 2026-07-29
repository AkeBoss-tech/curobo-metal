from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.graph_planning import (
    GraphPlanningProblem,
    PersistentRoadmap,
    paths_to_trajectory_seeds,
    plan_graph,
)
from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.optim import ExecutionCache
from curobo_metal.reference import (
    SerialRobot,
    problem_from_graph_dict,
    plan_graph as plan_reference,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "graph_planning"


def _problem(name: str, device: str = "cpu", dtype: torch.dtype = torch.float64):
    case = json.loads((FIXTURES / name).read_text())
    inputs, options = case["inputs"], case["options"]
    lo = torch.tensor(inputs["validity"]["lower"], dtype=dtype, device=device).reshape(
        -1, len(inputs["lower"])
    )
    hi = torch.tensor(inputs["validity"]["upper"], dtype=dtype, device=device).reshape_as(lo)

    def valid(q: torch.Tensor) -> torch.Tensor:
        if lo.shape[0] == 0:
            return torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
        inside = ((q[:, None] >= lo[None]) & (q[:, None] <= hi[None])).all(-1)
        return ~inside.any(-1)

    tensor = lambda value: torch.tensor(value, dtype=dtype, device=device)
    return case, GraphPlanningProblem(
        tensor(inputs["starts"]), tensor(inputs["goals"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]), validity=valid, **options,
    )


@pytest.mark.parametrize(
    "name", ["direct.json", "two_link_detour.json", "narrow_passage.json", "disconnected.json"]
)
def test_float64_cpu_exact_oracle_replay(name: str) -> None:
    case, problem = _problem(name)
    actual = plan_graph(problem)
    expected = plan_reference(problem_from_graph_dict(case))
    assert actual.status == expected.status
    assert actual.success.cpu().tolist() == expected.success.tolist()
    assert [vars(value) for value in actual.metrics] == [
        vars(value) for value in expected.metrics
    ]
    for torch_path, numpy_path in zip(actual.paths, expected.paths):
        np.testing.assert_allclose(torch_path.cpu().numpy(), numpy_path, rtol=0, atol=1e-15)
    for torch_path, numpy_path in zip(actual.roadmap_paths, expected.roadmap_paths):
        np.testing.assert_array_equal(torch_path.cpu().numpy(), numpy_path)


def test_batch_streams_are_independent_and_seed_handoff_is_fixed_knot() -> None:
    _, single = _problem("two_link_detour.json")
    one = plan_graph(single)
    batched = GraphPlanningProblem(
        **{
            **single.__dict__,
            "starts": single.starts.repeat(2, 1),
            "goals": single.goals.repeat(2, 1),
        }
    )
    many = plan_graph(batched)
    torch.testing.assert_close(many.paths[0], one.paths[0], rtol=0, atol=0)
    seeds = paths_to_trajectory_seeds(many, 17)
    assert seeds.shape == (2, 1, 17, 2)
    torch.testing.assert_close(seeds[:, 0, 0], batched.starts, rtol=0, atol=0)
    torch.testing.assert_close(seeds[:, 0, -1], batched.goals, rtol=0, atol=0)


def test_validation_errors_and_search_limit_status() -> None:
    _, base = _problem("two_link_detour.json")
    limited = GraphPlanningProblem(**{**base.__dict__, "max_search_expansions": 1})
    result = plan_graph(limited)
    assert result.status == ("search_limit",)
    with pytest.raises(ValueError, match="at least one connection"):
        plan_graph(GraphPlanningProblem(**{
            **base.__dict__, "k_neighbors": None, "connection_radius": None
        }))


def test_shape_keyed_sample_cache_reuse_and_reset() -> None:
    _, base = _problem("two_link_detour.json")
    cache = ExecutionCache(capacity=2)
    problem = GraphPlanningProblem(**{**base.__dict__, "execution_cache": cache})
    first, second = plan_graph(problem), plan_graph(problem)
    assert first.status == second.status
    assert cache.misses == 1 and cache.hits == 1 and cache.size == 1
    cache.reset()
    assert cache.size == 0
    third = plan_graph(problem)
    assert third.status == first.status and cache.misses == 2


def test_persistent_roadmap_sobol_greedy_lifecycle() -> None:
    _, base = _problem("direct.json")
    roadmap = PersistentRoadmap(capacity=2)
    problem = GraphPlanningProblem(**{
        **base.__dict__, "sampling": "sobol", "search": "greedy",
    })
    first = roadmap.plan(problem)
    second = roadmap.plan(problem)
    assert first.status == second.status
    assert roadmap.cache.hits == 1
    roadmap.reset()
    assert roadmap.generation == 1 and roadmap.cache.size == 0


def test_production_fk_and_collision_validate_swept_direct_edge() -> None:
    case = json.loads(
        (Path(__file__).parents[2] / "fixtures/trajectory/two_link_obstacle_detour.json").read_text()
    )
    inputs, raw = case["inputs"], case["inputs"]["collision"]
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    model = CollisionModel(
        tensor(raw["local_spheres"]),
        torch.tensor(raw["link_indices"], dtype=torch.int64),
        cuboid_centers=tensor(raw["cuboid_centers"]),
        cuboid_rotations=tensor(raw["cuboid_rotations"]),
        cuboid_half_extents=tensor(raw["cuboid_half_extents"]),
        padding=raw["padding"],
    )
    problem = GraphPlanningProblem(
        tensor(inputs["start"]), tensor(inputs["goal"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]),
        chain=KinematicChain(SerialRobot.from_dict(case["robot"]), dtype=torch.float64),
        collision_model=model, sample_count=0, k_neighbors=1, edge_step=0.02,
    )
    result = plan_graph(problem)
    assert result.status == ("disconnected",)
    assert result.metrics[0].edge_checks == 1
    assert result.metrics[0].validity_queries > 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_float32_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name in ("direct.json", "two_link_detour.json", "narrow_passage.json", "disconnected.json"):
        case, problem = _problem(name, "mps", torch.float32)
        result = plan_graph(problem)
        assert result.status == (case["expected"]["status"],)
        assert result.success.device.type == "mps"
        assert all(path.device.type == "mps" for path in result.paths)
