"""Deterministic geometric graph planning with device-batched validity checks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from typing import Callable, Sequence

import numpy as np
import torch

from curobo_metal.ops.costs import CollisionModel, robot_collision_cost
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.optim import ExecutionCache
from curobo_metal.ops.trajectory import (
    TrajectoryProblem,
    TrajectoryResult,
    optimize_trajectory,
)

ValidityCallback = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GraphPlanningProblem:
    """A batch of joint-space queries sharing limits and roadmap options."""

    starts: torch.Tensor
    goals: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    chain: KinematicChain | None = None
    collision_model: CollisionModel | Sequence[CollisionModel | None] | None = None
    validity: ValidityCallback | None = None
    collision_tolerance: float = 0.0
    sample_count: int = 128
    seed: int = 0
    k_neighbors: int | None = 12
    connection_radius: float | None = None
    edge_step: float = 0.05
    interpolation_step: float = 0.05
    search: str = "astar"
    shortcut: bool = True
    max_search_expansions: int | None = None
    execution_cache: ExecutionCache | None = None


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
    success: torch.Tensor
    status: tuple[str, ...]
    paths: tuple[torch.Tensor, ...]
    roadmap_paths: tuple[torch.Tensor, ...]
    metrics: tuple[PlanningMetrics, ...]


@dataclass(frozen=True)
class GraphTrajectoryResult:
    graph: GraphPlanningResult
    trajectory: TrajectoryResult | None


def _validate(problem: GraphPlanningProblem) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    values = (problem.starts, problem.goals, problem.lower, problem.upper)
    if not all(isinstance(x, torch.Tensor) for x in values):
        raise TypeError("starts, goals, and limits must be torch tensors")
    starts = problem.starts.unsqueeze(0) if problem.starts.ndim == 1 else problem.starts
    goals = problem.goals.unsqueeze(0) if problem.goals.ndim == 1 else problem.goals
    if starts.ndim != 2 or goals.shape != starts.shape or starts.shape[1] == 0:
        raise ValueError("starts and goals must have matching shape [J] or [B,J]")
    batch, dof = starts.shape
    if problem.lower.shape != (dof,) or problem.upper.shape != (dof,):
        raise ValueError(f"limits must have shape [{dof}]")
    reference = starts
    if any(x.device != reference.device or x.dtype != reference.dtype for x in values):
        raise ValueError("all planning tensors must have the same dtype and device")
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("planning tensors must use float32 or float64")
    if reference.device.type == "mps" and reference.dtype != torch.float32:
        raise TypeError("MPS graph planning supports only float32")
    if not all(bool(torch.isfinite(x).all().item()) for x in values):
        raise ValueError("planning tensors must contain only finite values")
    if bool((problem.lower >= problem.upper).any().item()):
        raise ValueError("every lower limit must be strictly less than its upper limit")
    if bool(((starts < problem.lower) | (starts > problem.upper)).any().item()):
        raise ValueError("starts must lie inside inclusive joint limits")
    if bool(((goals < problem.lower) | (goals > problem.upper)).any().item()):
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
        not math.isfinite(problem.connection_radius) or problem.connection_radius <= 0
    ):
        raise ValueError("connection_radius must be finite and positive or None")
    if problem.k_neighbors is None and problem.connection_radius is None:
        raise ValueError("at least one connection rule is required")
    for name, value in (
        ("edge_step", problem.edge_step),
        ("interpolation_step", problem.interpolation_step),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if problem.search not in ("astar", "dijkstra"):
        raise ValueError("search must be 'astar' or 'dijkstra'")
    if problem.max_search_expansions is not None and problem.max_search_expansions <= 0:
        raise ValueError("max_search_expansions must be positive or None")
    if not math.isfinite(problem.collision_tolerance) or problem.collision_tolerance < 0:
        raise ValueError("collision_tolerance must be finite and nonnegative")
    if problem.validity is None and problem.chain is None:
        raise ValueError("chain is required when validity is not supplied")
    if problem.chain is not None and (
        problem.chain.device != reference.device
        or problem.chain.dtype != reference.dtype
        or problem.chain.dof != dof
    ):
        raise ValueError("chain must match planning tensor device, dtype, and dof")
    if isinstance(problem.collision_model, Sequence) and len(problem.collision_model) != batch:
        raise ValueError("batched collision models must match problem batch")
    return starts, goals, batch, dof


def _model(problem: GraphPlanningProblem, batch_index: int) -> CollisionModel | None:
    if isinstance(problem.collision_model, Sequence):
        return problem.collision_model[batch_index]
    return problem.collision_model


def _valid(problem: GraphPlanningProblem, batch_index: int, points: torch.Tensor) -> torch.Tensor:
    if problem.validity is not None:
        value = problem.validity(points)
    else:
        model = _model(problem, batch_index)
        if model is None:
            value = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        else:
            transforms = forward_kinematics(problem.chain, points).transforms
            _, clearance = robot_collision_cost(transforms, model)
            value = clearance >= -problem.collision_tolerance
    if not isinstance(value, torch.Tensor):
        raise TypeError("validity callback must return a torch tensor")
    if value.device != points.device or value.dtype != torch.bool or value.shape != (points.shape[0],):
        raise ValueError("validity callback must return device-resident bool shape [N]")
    return value


def interpolate_edge(a: torch.Tensor, b: torch.Tensor, max_step: float) -> torch.Tensor:
    """Include endpoints with component-wise joint increments no larger than max_step."""
    count = max(1, int(math.ceil(float(torch.max(torch.abs(b - a)).item()) / max_step)))
    alpha = torch.linspace(0, 1, count + 1, dtype=a.dtype, device=a.device)
    return a[None] + alpha[:, None] * (b - a)[None]


def _sample(problem: GraphPlanningProblem, batch_index: int, dof: int) -> torch.Tensor:
    entry: dict[str, object] | None = None
    if problem.execution_cache is not None:
        entry = problem.execution_cache.acquire((
            "graph_samples", problem.sample_count, problem.seed, batch_index, dof,
            tuple(problem.lower.shape), str(problem.lower.dtype), problem.lower.device.type,
            tuple(float(x) for x in problem.lower.detach().cpu()),
            tuple(float(x) for x in problem.upper.detach().cpu()),
        ))
        cached = entry.get("samples")
        if isinstance(cached, torch.Tensor):
            return cached
    # PCG64 is deliberately generated on CPU to exactly preserve the replay contract.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [problem.seed, batch_index]
    )))
    unit = rng.random((problem.sample_count, dof), dtype=np.float64)
    lower = problem.lower.detach().cpu().double().numpy()
    upper = problem.upper.detach().cpu().double().numpy()
    samples = lower + unit * (upper - lower)
    result = torch.as_tensor(samples, dtype=problem.starts.dtype, device=problem.starts.device)
    if entry is not None:
        entry["samples"] = result
    return result


def _candidate_pairs(nodes_cpu: np.ndarray, problem: GraphPlanningProblem) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i in range(nodes_cpu.shape[0]):
        candidates = [
            (float(np.linalg.norm(nodes_cpu[j] - nodes_cpu[i])), j)
            for j in range(nodes_cpu.shape[0]) if j != i
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        if problem.connection_radius is not None:
            candidates = [item for item in candidates if item[0] <= problem.connection_radius]
        if problem.k_neighbors is not None:
            candidates = candidates[:problem.k_neighbors]
        pairs.update((min(i, j), max(i, j)) for _, j in candidates)
    return sorted(pairs)


def _validate_edges(
    problem: GraphPlanningProblem,
    batch_index: int,
    nodes: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> tuple[list[bool], int]:
    pieces = [interpolate_edge(nodes[i], nodes[j], problem.edge_step) for i, j in pairs]
    if not pieces:
        return [], 0
    lengths = [piece.shape[0] for piece in pieces]
    validity = _valid(problem, batch_index, torch.cat(pieces)).detach().cpu()
    return [
        bool(chunk.all().item()) for chunk in torch.split(validity, lengths)
    ], sum(lengths)


def _extract(
    nodes: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    search: str,
    limit: int | None,
) -> tuple[list[int] | None, int, bool]:
    distance = np.full(nodes.shape[0], np.inf)
    distance[0] = 0
    parent = np.full(nodes.shape[0], -1, dtype=np.int64)
    heuristic = float(np.linalg.norm(nodes[1] - nodes[0])) if search == "astar" else 0.0
    queue: list[tuple[float, float, int]] = [(heuristic, 0.0, 0)]
    closed = np.zeros(nodes.shape[0], dtype=np.bool_)
    expanded = 0
    while queue:
        _, cost, current = heapq.heappop(queue)
        if closed[current] or cost != distance[current]:
            continue
        closed[current] = True
        expanded += 1
        if limit is not None and expanded > limit:
            return None, expanded - 1, True
        if current == 1:
            path = [1]
            while path[-1] != 0:
                path.append(int(parent[path[-1]]))
            return path[::-1], expanded, False
        for neighbor, weight in adjacency[current]:
            candidate = cost + weight
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                parent[neighbor] = current
                h = float(np.linalg.norm(nodes[1] - nodes[neighbor])) if search == "astar" else 0.0
                heapq.heappush(queue, (candidate + h, candidate, neighbor))
    return None, expanded, False


def _shortcut(
    problem: GraphPlanningProblem, batch_index: int, path: torch.Tensor
) -> tuple[torch.Tensor, int, int, int]:
    if path.shape[0] <= 2:
        return path, 0, 0, 0
    selected, current = [0], 0
    checks = valids = queries = 0
    while current < path.shape[0] - 1:
        chosen = current + 1
        for candidate in range(path.shape[0] - 1, current, -1):
            outcomes, count = _validate_edges(
                problem, batch_index, path, [(current, candidate)]
            )
            queries += count
            checks += 1
            if outcomes[0]:
                chosen = candidate
                valids += 1
                break
        selected.append(chosen)
        current = chosen
    return path[selected], checks, valids, queries


def _dense(path: torch.Tensor, step: float) -> torch.Tensor:
    if path.shape[0] == 0:
        return path.clone()
    pieces = [interpolate_edge(path[i], path[i + 1], step)[:-1] for i in range(path.shape[0] - 1)]
    return torch.cat((*pieces, path[-1:]))


def plan_graph(problem: GraphPlanningProblem) -> GraphPlanningResult:
    """Plan independently per batch while batching node and edge validity on device."""
    starts, goals, batch, dof = _validate(problem)
    success = torch.zeros(batch, dtype=torch.bool, device=starts.device)
    statuses, paths, roadmap_paths, metrics = [], [], [], []
    for b in range(batch):
        endpoints = _valid(problem, b, torch.stack((starts[b], goals[b]))).detach().cpu()
        queries = 2
        if not bool(endpoints.all().item()):
            status = "invalid_start" if not bool(endpoints[0]) else "invalid_goal"
            empty = starts.new_empty((0, dof))
            statuses.append(status); paths.append(empty); roadmap_paths.append(empty.clone())
            metrics.append(PlanningMetrics(problem.sample_count, 0, queries, 0, 0, 0, math.inf))
            continue
        samples = _sample(problem, b, dof)
        mask = _valid(problem, b, samples)
        queries += samples.shape[0]
        valid_samples = samples[mask]
        nodes = torch.cat((starts[b:b + 1], goals[b:b + 1], valid_samples))
        nodes_cpu = nodes.detach().cpu().double().numpy()
        pairs = _candidate_pairs(nodes_cpu, problem)
        edge_valid, edge_queries = _validate_edges(problem, b, nodes, pairs)
        queries += edge_queries
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(nodes.shape[0])]
        edges_valid = 0
        for (i, j), valid in zip(pairs, edge_valid):
            if valid:
                weight = float(np.linalg.norm(nodes_cpu[j] - nodes_cpu[i]))
                adjacency[i].append((j, weight)); adjacency[j].append((i, weight))
                edges_valid += 1
        for neighbors in adjacency:
            neighbors.sort(key=lambda item: item[0])
        indices, expanded, limited = _extract(
            nodes_cpu, adjacency, problem.search, problem.max_search_expansions
        )
        if indices is None:
            empty = starts.new_empty((0, dof))
            statuses.append("search_limit" if limited else "disconnected")
            paths.append(empty); roadmap_paths.append(empty.clone())
            metrics.append(PlanningMetrics(
                problem.sample_count, valid_samples.shape[0], queries, len(pairs),
                edges_valid, expanded, math.inf,
            ))
            continue
        roadmap = nodes[indices]
        sparse = roadmap
        checks = shortcut_valids = shortcut_queries = 0
        if problem.shortcut:
            sparse, checks, shortcut_valids, shortcut_queries = _shortcut(problem, b, roadmap)
        queries += shortcut_queries
        dense = _dense(sparse, min(problem.interpolation_step, problem.edge_step))
        sparse_cpu = sparse.detach().cpu().double().numpy()
        cost = float(np.linalg.norm(np.diff(sparse_cpu, axis=0), axis=1).sum())
        success[b] = True
        statuses.append("direct_success" if sparse.shape[0] == 2 else "success")
        paths.append(dense); roadmap_paths.append(roadmap)
        metrics.append(PlanningMetrics(
            problem.sample_count, valid_samples.shape[0], queries, len(pairs) + checks,
            edges_valid + shortcut_valids, expanded, cost,
        ))
    return GraphPlanningResult(success, tuple(statuses), tuple(paths), tuple(roadmap_paths), tuple(metrics))


def paths_to_trajectory_seeds(
    result: GraphPlanningResult, steps: int
) -> torch.Tensor:
    """Interpolate successful graph paths to fixed-knot trajectory seeds [B,1,T,J]."""
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    if not result.paths:
        raise ValueError("graph result must contain at least one batch item")
    if not bool(result.success.all().item()):
        raise ValueError("every graph query must succeed before trajectory seed handoff")
    rows = []
    for path in result.paths:
        segment = torch.linalg.vector_norm(torch.diff(path, dim=0), dim=1)
        cumulative = torch.cat((segment.new_zeros(1), torch.cumsum(segment, 0)))
        targets = torch.linspace(0, float(cumulative[-1].item()), steps, device=path.device, dtype=path.dtype)
        indices = torch.searchsorted(cumulative, targets, right=True).clamp(1, path.shape[0] - 1)
        lo, hi = cumulative[indices - 1], cumulative[indices]
        alpha = ((targets - lo) / (hi - lo).clamp_min(torch.finfo(path.dtype).eps))[:, None]
        seed = path[indices - 1] + alpha * (path[indices] - path[indices - 1])
        seed[0], seed[-1] = path[0], path[-1]
        rows.append(seed)
    return torch.stack(rows)[:, None]


def plan_and_optimize(
    graph_problem: GraphPlanningProblem,
    trajectory_problem: TrajectoryProblem,
    *,
    differentiable: bool = False,
) -> GraphTrajectoryResult:
    """Plan geometric seeds, then hand them to production trajectory optimization."""
    graph = plan_graph(graph_problem)
    if not bool(graph.success.all().item()):
        return GraphTrajectoryResult(graph, None)
    seeds = paths_to_trajectory_seeds(graph, trajectory_problem.steps)
    if trajectory_problem.start.ndim == 1:
        seeds = seeds[0]
    trajectory = optimize_trajectory(
        replace(trajectory_problem, seeds=seeds), differentiable=differentiable
    )
    return GraphTrajectoryResult(graph, trajectory)
