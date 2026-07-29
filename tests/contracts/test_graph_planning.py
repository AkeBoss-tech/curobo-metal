from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    CollisionModel,
    GraphPlanningProblem,
    SerialRobot,
    interpolate_edge,
    load_case,
    load_graph_planning_case,
    plan_graph,
    problem_from_graph_dict,
    robot_collision_validity,
    save_graph_planning_case,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
GRAPH_FIXTURES = FIXTURES / "graph_planning"


@pytest.mark.parametrize(
    "name", ["direct.json", "two_link_detour.json", "narrow_passage.json",
             "disconnected.json"]
)
def test_canonical_replays_and_independent_path_evidence(name: str) -> None:
    path = GRAPH_FIXTURES / name
    case = load_graph_planning_case(path)
    assert path.read_text() == (
        json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    problem = problem_from_graph_dict(case)
    result = plan_graph(problem)
    expected = case["expected"]
    assert result.status[0] == expected["status"]
    assert bool(result.success[0]) is expected["success"]
    if expected["success"]:
        actual_path = result.paths[0]
        np.testing.assert_array_equal(actual_path[0], problem.starts[0])
        np.testing.assert_array_equal(actual_path[-1], problem.goals[0])
        # Re-evaluate every returned swept-validation/interpolation point.
        assert np.all(problem.validity(actual_path))
        assert np.max(np.abs(np.diff(actual_path, axis=0))) <= problem.edge_step + 1e-15
        independent_cost = float(np.linalg.norm(np.diff(actual_path, axis=0), axis=1).sum())
        assert independent_cost == pytest.approx(expected["path_cost"], abs=1e-12)
        assert result.metrics[0].path_cost == pytest.approx(independent_cost, abs=1e-12)
    else:
        assert result.paths[0].shape == (0, problem.starts.shape[1])
        assert np.isinf(result.metrics[0].path_cost)


def test_detour_avoids_box_and_has_necessary_extra_cost() -> None:
    case = load_graph_planning_case(GRAPH_FIXTURES / "two_link_detour.json")
    problem = problem_from_graph_dict(case)
    result = plan_graph(problem)
    path = result.paths[0]
    assert np.any(np.abs(path[:, 1]) > 1.1)
    straight_cost = np.linalg.norm(problem.goals[0] - problem.starts[0])
    assert result.metrics[0].path_cost > straight_cost + 0.5


def test_deterministic_batch_sampling_and_batch_index_streams() -> None:
    validity = lambda q: np.ones(q.shape[0], dtype=np.bool_)
    kwargs = dict(
        starts=np.array([[-1.0, 0.0], [-1.0, 0.0]]),
        goals=np.array([[1.0, 0.0], [1.0, 0.0]]),
        lower=np.full(2, -2.0), upper=np.full(2, 2.0), validity=validity,
        sample_count=32, seed=93, k_neighbors=8, edge_step=0.1,
    )
    first = plan_graph(GraphPlanningProblem(**kwargs))
    second = plan_graph(GraphPlanningProblem(**kwargs))
    for a, b in zip(first.roadmap_paths, second.roadmap_paths):
        np.testing.assert_array_equal(a, b)
    assert first.metrics == second.metrics
    # Identical queries use independent deterministic batch-index streams.
    assert not np.array_equal(first.roadmap_paths[0], first.roadmap_paths[1])


def test_swept_edges_reject_valid_endpoints_across_thin_obstacle() -> None:
    def validity(q: np.ndarray) -> np.ndarray:
        return ~((np.abs(q[:, 0]) <= 0.01) & (np.abs(q[:, 1]) <= 1.0))

    problem = GraphPlanningProblem(
        starts=np.array([[-1.0, 0.0]]), goals=np.array([[1.0, 0.0]]),
        lower=np.full(2, -1.5), upper=np.full(2, 1.5), validity=validity,
        sample_count=0, k_neighbors=2, edge_step=0.01,
    )
    result = plan_graph(problem)
    assert result.status == ("disconnected",)
    assert result.metrics[0].edge_checks == 1
    assert result.metrics[0].edges_valid == 0


def test_endpoint_failures_search_limit_and_validation() -> None:
    invalid = lambda q: np.zeros(q.shape[0], dtype=np.bool_)
    base = dict(
        starts=np.array([[0.0, 0.0]]), goals=np.array([[1.0, 1.0]]),
        lower=np.full(2, -2.0), upper=np.full(2, 2.0), validity=invalid,
    )
    assert plan_graph(GraphPlanningProblem(**base)).status == ("invalid_start",)
    valid = lambda q: np.ones(q.shape[0], dtype=np.bool_)
    limited = plan_graph(GraphPlanningProblem(
        **dict(base, validity=valid), sample_count=8, k_neighbors=4,
        max_search_expansions=1,
    ))
    assert limited.status == ("search_limit",)
    with pytest.raises(TypeError):
        plan_graph(GraphPlanningProblem(**dict(base, starts=np.zeros((1, 2), dtype=int))))
    with pytest.raises(ValueError):
        plan_graph(GraphPlanningProblem(**dict(base, validity=valid), edge_step=0.0))
    with pytest.raises(ValueError, match="callback"):
        plan_graph(GraphPlanningProblem(
            **dict(base, validity=lambda q: np.ones((q.shape[0], 1), dtype=np.bool_))
        ))
    empty = plan_graph(GraphPlanningProblem(
        starts=np.empty((0, 2)), goals=np.empty((0, 2)),
        lower=np.full(2, -1.0), upper=np.full(2, 1.0), validity=valid,
    ))
    assert empty.success.shape == (0,)
    assert empty.paths == () and empty.metrics == ()


def test_existing_fk_collision_composes_as_validity_callback() -> None:
    robot = SerialRobot.from_dict(load_case(FIXTURES / "two_link_planar.json")["robot"])
    model = CollisionModel(
        local_spheres=np.array([[0.0, 0.0, 0.0, 0.1]]),
        link_indices=np.array([2]),
        cuboid_centers=np.array([[1.75, 0.0, 0.0]]),
        cuboid_rotations=np.eye(3)[None],
        cuboid_half_extents=np.array([[0.05, 0.05, 0.05]]),
        activation_distance=0.0,
        weight=2.0,
    )
    callback = robot_collision_validity(robot, model)
    assert callback(np.array([[0.0, 0.0], [np.pi / 2, 0.0]])).tolist() == [False, True]


def test_panda_class_direct_query_and_no_gpu_dependency(tmp_path: Path) -> None:
    robot = SerialRobot.from_dict(load_case(FIXTURES / "panda_serial.json")["robot"])
    valid = lambda q: np.ones(q.shape[0], dtype=np.bool_)
    start = np.zeros((1, robot.dof), dtype=np.float64)
    goal = np.full((1, robot.dof), 0.2, dtype=np.float64)
    result = plan_graph(GraphPlanningProblem(
        start, goal, np.full(robot.dof, -3.0), np.full(robot.dof, 3.0), valid,
        sample_count=24, seed=7, k_neighbors=8, edge_step=0.05,
    ))
    assert result.success.tolist() == [True]
    np.testing.assert_allclose(result.paths[0][-1], goal[0], atol=0, rtol=0)
    case = load_graph_planning_case(GRAPH_FIXTURES / "direct.json")
    output = tmp_path / "case.json"
    save_graph_planning_case(output, case)
    assert output.read_bytes() == (GRAPH_FIXTURES / "direct.json").read_bytes()
    module = importlib.import_module("curobo_metal.reference.graph_planning")
    source = Path(module.__file__).read_text()
    for forbidden in ("import torch", "import curobo", "import warp", "import isaac"):
        assert forbidden not in source
