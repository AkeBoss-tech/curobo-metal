"""Deterministic, NumPy-only joint-space graph-planning oracle."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .costs import CollisionModel, robot_collision_cost
from .forward_kinematics import SerialRobot

FloatArray = NDArray[np.float64]
ValidityCallback = Callable[[FloatArray], NDArray[np.bool_]]

GRAPH_PLANNING_FORMAT = "curobo-metal-graph-planning-case"
GRAPH_PLANNING_VERSION = 1


@dataclass(frozen=True)
class GraphPlanningProblem:
    """One batched planning query; batches share limits and planner options."""

    starts: FloatArray
    goals: FloatArray
    lower: FloatArray
    upper: FloatArray
    validity: ValidityCallback
    sample_count: int = 128
    seed: int = 0
    k_neighbors: int | None = 12
    connection_radius: float | None = None
    edge_step: float = 0.05
    interpolation_step: float = 0.05
    search: str = "astar"
    shortcut: bool = True
    max_search_expansions: int | None = None


@dataclass(frozen=True)
class PlanningMetrics:
    samples_requested: int
    samples_valid: int
    validity_queries: int
    edge_checks: int
    edges_valid: int
    nodes_expanded: int
    path_cost: float


@dataclass(frozen=True)
class GraphPlanningResult:
    success: NDArray[np.bool_]
    status: tuple[str, ...]
    paths: tuple[FloatArray, ...]
    roadmap_paths: tuple[FloatArray, ...]
    metrics: tuple[PlanningMetrics, ...]


def _validate(problem: GraphPlanningProblem) -> tuple[int, int]:
    arrays = (problem.starts, problem.goals, problem.lower, problem.upper)
    if not all(isinstance(x, np.ndarray) and x.dtype.kind == "f" for x in arrays):
        raise TypeError("starts, goals, and limits must be floating NumPy arrays")
    if problem.starts.ndim != 2 or problem.goals.shape != problem.starts.shape:
        raise ValueError("starts and goals must have matching shape [B,J]")
    batch, dof = problem.starts.shape
    if dof == 0:
        raise ValueError("planning requires at least one joint")
    if problem.lower.shape != (dof,) or problem.upper.shape != (dof,):
        raise ValueError(f"limits must have shape [{dof}]")
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("planning arrays must contain only finite values")
    if np.any(problem.lower >= problem.upper):
        raise ValueError("every lower limit must be strictly less than its upper limit")
    if np.any(problem.starts < problem.lower) or np.any(problem.starts > problem.upper):
        raise ValueError("starts must lie inside inclusive joint limits")
    if np.any(problem.goals < problem.lower) or np.any(problem.goals > problem.upper):
        raise ValueError("goals must lie inside inclusive joint limits")
    if not isinstance(problem.sample_count, int) or problem.sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    if not isinstance(problem.seed, int) or problem.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if problem.k_neighbors is not None and (
        not isinstance(problem.k_neighbors, int) or problem.k_neighbors <= 0
    ):
        raise ValueError("k_neighbors must be positive or None")
    if problem.connection_radius is not None and (
        not np.isfinite(problem.connection_radius) or problem.connection_radius <= 0
    ):
        raise ValueError("connection_radius must be finite and positive or None")
    if problem.k_neighbors is None and problem.connection_radius is None:
        raise ValueError("at least one connection rule is required")
    if not np.isfinite(problem.edge_step) or problem.edge_step <= 0:
        raise ValueError("edge_step must be finite and positive")
    if not np.isfinite(problem.interpolation_step) or problem.interpolation_step <= 0:
        raise ValueError("interpolation_step must be finite and positive")
    if problem.search not in ("astar", "dijkstra"):
        raise ValueError("search must be 'astar' or 'dijkstra'")
    if problem.max_search_expansions is not None and problem.max_search_expansions <= 0:
        raise ValueError("max_search_expansions must be positive or None")
    return batch, dof


def _valid(callback: ValidityCallback, points: FloatArray) -> NDArray[np.bool_]:
    value = np.asarray(callback(np.asarray(points, dtype=np.float64)))
    if value.dtype != np.bool_ or value.shape != (points.shape[0],):
        raise ValueError("validity callback must return bool shape [N]")
    return value


def interpolate_edge(a: FloatArray, b: FloatArray, max_step: float) -> FloatArray:
    """Include both endpoints with component-wise joint increments <= max_step."""
    count = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / max_step)))
    return a[None] + np.linspace(0.0, 1.0, count + 1)[:, None] * (b - a)[None]


def _edge_valid(
    callback: ValidityCallback, a: FloatArray, b: FloatArray, step: float
) -> tuple[bool, int]:
    points = interpolate_edge(a, b, step)
    return bool(np.all(_valid(callback, points))), points.shape[0]


def _sample(problem: GraphPlanningProblem, batch_index: int, dof: int) -> FloatArray:
    # A per-batch SeedSequence makes a batch invariant to the presence of other batches.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [problem.seed, batch_index]
    )))
    unit = rng.random((problem.sample_count, dof), dtype=np.float64)
    return problem.lower + unit * (problem.upper - problem.lower)


def _extract(
    nodes: FloatArray,
    adjacency: list[list[tuple[int, float]]],
    search: str,
    expansion_limit: int | None,
) -> tuple[list[int] | None, int, bool]:
    goal = 1
    distance = np.full(nodes.shape[0], np.inf)
    distance[0] = 0.0
    parent = np.full(nodes.shape[0], -1, dtype=np.int64)
    queue: list[tuple[float, float, int]] = [(float(np.linalg.norm(nodes[goal] - nodes[0]))
                                               if search == "astar" else 0.0, 0.0, 0)]
    expanded = 0
    closed = np.zeros(nodes.shape[0], dtype=np.bool_)
    while queue:
        _, cost, current = heapq.heappop(queue)
        if closed[current] or cost != distance[current]:
            continue
        closed[current] = True
        expanded += 1
        if expansion_limit is not None and expanded > expansion_limit:
            return None, expanded - 1, True
        if current == goal:
            path = [goal]
            while path[-1] != 0:
                path.append(int(parent[path[-1]]))
            return path[::-1], expanded, False
        for neighbor, weight in adjacency[current]:
            candidate = cost + weight
            # Strict improvement plus serialized node order gives deterministic ties.
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                parent[neighbor] = current
                heuristic = (
                    float(np.linalg.norm(nodes[goal] - nodes[neighbor]))
                    if search == "astar" else 0.0
                )
                heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
    return None, expanded, False


def _shortcut(
    path: FloatArray, callback: ValidityCallback, edge_step: float
) -> tuple[FloatArray, int, int, int]:
    if path.shape[0] <= 2:
        return path, 0, 0, 0
    selected = [0]
    checks = valid_edges = queries = 0
    current = 0
    while current < path.shape[0] - 1:
        chosen = current + 1
        for candidate in range(path.shape[0] - 1, current, -1):
            ok, count = _edge_valid(callback, path[current], path[candidate], edge_step)
            checks += 1
            queries += count
            if ok:
                chosen = candidate
                valid_edges += 1
                break
        selected.append(chosen)
        current = chosen
    return path[selected], checks, valid_edges, queries


def _dense_path(path: FloatArray, step: float) -> FloatArray:
    if path.shape[0] == 0:
        return path.copy()
    pieces = [interpolate_edge(path[i], path[i + 1], step)[:-1]
              for i in range(path.shape[0] - 1)]
    pieces.append(path[-1:])
    return np.concatenate(pieces, axis=0)


def plan_graph(problem: GraphPlanningProblem) -> GraphPlanningResult:
    """Build independent deterministic roadmaps and solve every batch item."""
    batch, dof = _validate(problem)
    successes = np.zeros(batch, dtype=np.bool_)
    statuses: list[str] = []
    paths: list[FloatArray] = []
    roadmap_paths: list[FloatArray] = []
    all_metrics: list[PlanningMetrics] = []
    for b in range(batch):
        endpoint_validity = _valid(problem.validity, np.stack((problem.starts[b], problem.goals[b])))
        validity_queries = 2
        if not endpoint_validity[0] or not endpoint_validity[1]:
            status = "invalid_start" if not endpoint_validity[0] else "invalid_goal"
            empty = np.empty((0, dof), dtype=np.float64)
            statuses.append(status)
            paths.append(empty)
            roadmap_paths.append(empty.copy())
            all_metrics.append(PlanningMetrics(
                problem.sample_count, 0, validity_queries, 0, 0, 0, np.inf
            ))
            continue
        samples = _sample(problem, b, dof)
        sample_mask = _valid(problem.validity, samples)
        validity_queries += samples.shape[0]
        valid_samples = samples[sample_mask]
        nodes = np.concatenate((problem.starts[b:b + 1], problem.goals[b:b + 1],
                                valid_samples), axis=0)
        count = nodes.shape[0]
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
        edge_checks = edges_valid = 0
        candidate_pairs: set[tuple[int, int]] = set()
        for i in range(count):
            candidates = [
                (float(np.linalg.norm(nodes[j] - nodes[i])), j)
                for j in range(count) if j != i
            ]
            candidates.sort(key=lambda item: (item[0], item[1]))
            if problem.connection_radius is not None:
                candidates = [
                    item for item in candidates
                    if item[0] <= problem.connection_radius
                ]
            if problem.k_neighbors is not None:
                candidates = candidates[:problem.k_neighbors]
            candidate_pairs.update((min(i, j), max(i, j)) for _, j in candidates)
        for i, j in sorted(candidate_pairs):
            distance = float(np.linalg.norm(nodes[j] - nodes[i]))
            ok, queried = _edge_valid(problem.validity, nodes[i], nodes[j],
                                      problem.edge_step)
            edge_checks += 1
            validity_queries += queried
            if ok:
                edges_valid += 1
                adjacency[i].append((j, distance))
                adjacency[j].append((i, distance))
        for neighbors in adjacency:
            neighbors.sort(key=lambda item: item[0])
        indices, expanded, limited = _extract(
            nodes, adjacency, problem.search, problem.max_search_expansions
        )
        if indices is None:
            status = "search_limit" if limited else "disconnected"
            empty = np.empty((0, dof), dtype=np.float64)
            statuses.append(status)
            paths.append(empty)
            roadmap_paths.append(empty.copy())
            all_metrics.append(PlanningMetrics(
                problem.sample_count, valid_samples.shape[0], validity_queries,
                edge_checks, edges_valid, expanded, np.inf
            ))
            continue
        roadmap = nodes[indices]
        sparse = roadmap
        if problem.shortcut:
            sparse, checks, valids, queries = _shortcut(
                roadmap, problem.validity, problem.edge_step
            )
            edge_checks += checks
            edges_valid += valids
            validity_queries += queries
        # Preserve every swept-validation sample in the returned path.  A coarser
        # requested output step must never introduce unvalidated waypoints.
        dense = _dense_path(sparse, min(problem.interpolation_step, problem.edge_step))
        cost = float(np.sum(np.linalg.norm(np.diff(sparse, axis=0), axis=1)))
        successes[b] = True
        statuses.append("direct_success" if sparse.shape[0] == 2 else "success")
        paths.append(dense)
        roadmap_paths.append(roadmap)
        all_metrics.append(PlanningMetrics(
            problem.sample_count, valid_samples.shape[0], validity_queries,
            edge_checks, edges_valid, expanded, cost
        ))
    return GraphPlanningResult(
        successes, tuple(statuses), tuple(paths), tuple(roadmap_paths), tuple(all_metrics)
    )


def joint_box_validity(lower: FloatArray, upper: FloatArray) -> ValidityCallback:
    """Return validity outside a union of closed forbidden joint-space boxes."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 2 or hi.shape != lo.shape or np.any(lo > hi):
        raise ValueError("forbidden box bounds must match [K,J] with lower <= upper")

    def callback(points: FloatArray) -> NDArray[np.bool_]:
        q = np.asarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != lo.shape[1]:
            raise ValueError("validity points must have shape [N,J]")
        inside = np.all((q[:, None, :] >= lo[None]) & (q[:, None, :] <= hi[None]), axis=2)
        return ~np.any(inside, axis=1)

    return callback


def robot_collision_validity(
    robot: SerialRobot, collision_model: CollisionModel, tolerance: float = 0.0
) -> ValidityCallback:
    """Compose existing FK/collision references into a batched validity callback."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    def callback(points: FloatArray) -> NDArray[np.bool_]:
        q = np.asarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != robot.dof:
            raise ValueError(f"validity points must have shape [N,{robot.dof}]")
        return np.asarray(
            [robot_collision_cost(robot, point, collision_model) <= tolerance for point in q],
            dtype=np.bool_,
        )

    return callback


def problem_from_graph_dict(value: Mapping[str, Any]) -> GraphPlanningProblem:
    inputs = value["inputs"]
    validity = inputs["validity"]
    if validity.get("type") != "joint_boxes":
        raise ValueError("replay validity type must be 'joint_boxes'")
    dof = len(inputs["lower"])
    box_lower = np.asarray(validity["lower"], dtype=np.float64).reshape((-1, dof))
    box_upper = np.asarray(validity["upper"], dtype=np.float64).reshape((-1, dof))
    options = value.get("options", {})
    return GraphPlanningProblem(
        starts=np.asarray(inputs["starts"], dtype=np.float64),
        goals=np.asarray(inputs["goals"], dtype=np.float64),
        lower=np.asarray(inputs["lower"], dtype=np.float64),
        upper=np.asarray(inputs["upper"], dtype=np.float64),
        validity=joint_box_validity(box_lower, box_upper),
        sample_count=int(options.get("sample_count", 128)),
        seed=int(options.get("seed", 0)),
        k_neighbors=options.get("k_neighbors", 12),
        connection_radius=options.get("connection_radius"),
        edge_step=float(options.get("edge_step", 0.05)),
        interpolation_step=float(options.get("interpolation_step", 0.05)),
        search=str(options.get("search", "astar")),
        shortcut=bool(options.get("shortcut", True)),
        max_search_expansions=options.get("max_search_expansions"),
    )


def load_graph_planning_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if (value.get("format") != GRAPH_PLANNING_FORMAT
            or value.get("version") != GRAPH_PLANNING_VERSION):
        raise ValueError("unsupported graph-planning replay format")
    return value


def save_graph_planning_case(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_graph_planning_case(destination)
