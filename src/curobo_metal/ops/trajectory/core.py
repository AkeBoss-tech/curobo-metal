"""Production batched trajectory costs and projected optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from curobo_metal.optim import (
    ExecutionCache, LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize,
)
from curobo_metal.ops.costs import CollisionModel, robot_collision_cost
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics

TrajectoryStatus = str


@dataclass(frozen=True)
class TrajectoryWeights:
    endpoint: float = 1_000.0
    joint_limit: float = 10.0
    velocity: float = 0.05
    acceleration: float = 1.0
    jerk: float = 0.05


@dataclass(frozen=True)
class TrajectoryProblem:
    chain: KinematicChain
    start: torch.Tensor
    goal: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    steps: int
    dt: float
    seeds: torch.Tensor | None = None
    weights: TrajectoryWeights = TrajectoryWeights()
    collision_model: CollisionModel | Sequence[CollisionModel | None] | None = None
    collision_subdivisions: int = 1
    endpoint_tolerance: float = 1e-4
    collision_tolerance: float = 0.0
    max_iterations: int = 200
    gradient_tolerance: float = 1e-7
    learning_rate: float = 0.03
    optimizer: str = "adam"
    lbfgs: LBFGSConfig | None = None
    particle: ParticleConfig | None = None
    optimizer_cache: ExecutionCache | None = None
    warm_start: bool = False


@dataclass(frozen=True)
class TrajectoryCost:
    value: torch.Tensor
    endpoint: torch.Tensor
    joint_limit: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    collision: torch.Tensor
    minimum_clearance: torch.Tensor


@dataclass(frozen=True)
class TrajectoryMetrics:
    duration: float
    path_length: torch.Tensor
    maximum_velocity: torch.Tensor
    maximum_acceleration: torch.Tensor
    maximum_jerk: torch.Tensor
    minimum_clearance: torch.Tensor
    maximum_limit_violation: torch.Tensor


@dataclass(frozen=True)
class TrajectoryResult:
    trajectories: torch.Tensor
    success: torch.Tensor
    status: tuple[TrajectoryStatus, ...] | tuple[tuple[TrajectoryStatus, ...], ...]
    iterations: torch.Tensor
    objective: torch.Tensor
    endpoint_error: torch.Tensor
    minimum_clearance: torch.Tensor
    maximum_limit_violation: torch.Tensor
    selected_seed: int | None | tuple[int | None, ...]
    input_problems_were_batched: bool


def _floating(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be float32 or float64")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS trajectory operations support only float32")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def minimum_jerk_trajectory(
    start: torch.Tensor, goal: torch.Tensor, steps: int
) -> torch.Tensor:
    """Quintic interpolation for ``[J]`` or ``[B,J]`` endpoints."""
    q0, q1 = _floating(start, "start"), _floating(goal, "goal")
    if q0.shape != q1.shape or q0.ndim not in (1, 2):
        raise ValueError("start and goal must have equal shape [J] or [B,J]")
    if q0.device != q1.device or q0.dtype != q1.dtype:
        raise ValueError("start and goal must share dtype and device")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    phase = torch.linspace(0, 1, steps, dtype=q0.dtype, device=q0.device)
    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    return q0.unsqueeze(-2) + blend[:, None] * (q1 - q0).unsqueeze(-2)


def interpolated_states(
    q: torch.Tensor, subdivisions: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled states and knot coordinates, preserving leading batches."""
    trajectory = _floating(q, "q")
    if trajectory.ndim < 2 or trajectory.shape[-2] < 2:
        raise ValueError("q must end in [T,J] with T >= 2")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    count = (trajectory.shape[-2] - 1) * subdivisions + 1
    coordinates = torch.arange(count, device=q.device, dtype=q.dtype) / subdivisions
    low = coordinates.to(torch.int64).clamp_max(trajectory.shape[-2] - 2)
    fraction = coordinates - low
    fraction[-1] = 1
    states = (
        (1 - fraction).reshape((1,) * (q.ndim - 2) + (-1, 1)) * trajectory[..., low, :]
        + fraction.reshape((1,) * (q.ndim - 2) + (-1, 1)) * trajectory[..., low + 1, :]
    )
    return states, coordinates


def endpoint_cost(
    q: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, *, weight: float = 1.0
) -> torch.Tensor:
    return 0.5 * weight * (
        (q[..., 0, :] - start).square().sum(-1)
        + (q[..., -1, :] - goal).square().sum(-1)
    )


def joint_limit_cost(
    q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, *, weight: float = 1.0
) -> torch.Tensor:
    return 0.5 * weight * (
        (lower - q).clamp_min(0).square() + (q - upper).clamp_min(0).square()
    ).sum(dim=(-2, -1))


def derivative_cost(q: torch.Tensor, order: int, dt: float, *, weight: float = 1.0) -> torch.Tensor:
    """Physical-time rectangular integral of a finite-difference derivative."""
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(weight) or weight < 0:
        raise ValueError("dt must be positive and weight finite and nonnegative")
    if q.shape[-2] <= order:
        return q.sum(dim=(-2, -1)) * 0
    difference = torch.diff(q, n=order, dim=-2)
    return 0.5 * weight * dt ** (1 - 2 * order) * difference.square().sum((-2, -1))


def trajectory_collision_cost(
    chain: KinematicChain,
    q: torch.Tensor,
    model: CollisionModel | None,
    *,
    subdivisions: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collision hinge and minimum clearance for every leading trajectory item."""
    leading = q.shape[:-2]
    states, _ = interpolated_states(q, subdivisions)
    flat = states.reshape(-1, q.shape[-1])
    if model is None:
        zero = q.sum(dim=(-2, -1)) * 0
        return zero, q.new_full(leading, torch.inf)
    transforms = forward_kinematics(chain, flat).transforms
    sample_cost, clearance = robot_collision_cost(transforms, model)
    sample_count = states.shape[-2]
    return (
        sample_cost.reshape(leading + (sample_count,)).sum(-1),
        clearance.reshape(leading + (sample_count,)).min(-1).values,
    )


def _models(problem: TrajectoryProblem, batch: int) -> tuple[CollisionModel | None, ...]:
    model = problem.collision_model
    if isinstance(model, Sequence):
        if len(model) != batch:
            raise ValueError("batched collision models must match problem batch")
        return tuple(model)
    return (model,) * batch


def evaluate_trajectory(problem: TrajectoryProblem, q: torch.Tensor) -> TrajectoryCost:
    """Evaluate costs for normalized ``[B,N,T,J]`` trajectories."""
    start, goal = problem.start, problem.goal
    if start.ndim == 1:
        start, goal = start[None], goal[None]
    if q.ndim == 3:
        q = q[None]
    if q.ndim != 4:
        raise ValueError("q must have shape [N,T,J] or [B,N,T,J]")
    starts, goals = start[:, None], goal[:, None]
    lower, upper = problem.lower, problem.upper
    if lower.ndim == 1:
        lower, upper = lower[None], upper[None]
    lower, upper = lower[:, None, None], upper[:, None, None]
    end = endpoint_cost(q, starts, goals, weight=problem.weights.endpoint)
    limit = joint_limit_cost(q, lower, upper, weight=problem.weights.joint_limit)
    velocity = derivative_cost(q, 1, problem.dt, weight=problem.weights.velocity)
    acceleration = derivative_cost(q, 2, problem.dt, weight=problem.weights.acceleration)
    jerk = derivative_cost(q, 3, problem.dt, weight=problem.weights.jerk)
    collision_rows, clearance_rows = [], []
    for batch_index, model in enumerate(_models(problem, q.shape[0])):
        cost, clearance = trajectory_collision_cost(
            problem.chain, q[batch_index], model,
            subdivisions=problem.collision_subdivisions,
        )
        collision_rows.append(cost)
        clearance_rows.append(clearance)
    collision = torch.stack(collision_rows)
    clearance = torch.stack(clearance_rows)
    return TrajectoryCost(
        end + limit + velocity + acceleration + jerk + collision,
        end, limit, velocity, acceleration, jerk, collision, clearance,
    )


def _validate(problem: TrajectoryProblem) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if not isinstance(problem.chain, KinematicChain):
        raise TypeError("chain must be a KinematicChain")
    start, goal = _floating(problem.start, "start"), _floating(problem.goal, "goal")
    lower, upper = _floating(problem.lower, "lower"), _floating(problem.upper, "upper")
    batched = start.ndim == 2
    if start.ndim == 1:
        start, goal = start[None], goal[None]
    if start.ndim != 2 or goal.shape != start.shape or start.shape[1] != problem.chain.dof:
        raise ValueError("start and goal must have equal shape [J] or [B,J]")
    if lower.ndim == 1:
        lower, upper = lower[None].expand_as(start), upper[None].expand_as(start)
    if lower.shape != start.shape or upper.shape != start.shape:
        raise ValueError("limits must have shape [J] or [B,J]")
    for value in (start, goal, lower, upper):
        if value.device.type != problem.chain.device.type or value.dtype != problem.chain.dtype:
            raise ValueError("trajectory tensors must match chain dtype and device")
    if bool((lower > upper).any().item()):
        raise ValueError("lower must not exceed upper")
    if problem.steps < 2 or problem.collision_subdivisions < 1 or problem.max_iterations <= 0:
        raise ValueError("steps, subdivisions, and max_iterations must be positive")
    scalars = (problem.dt, problem.endpoint_tolerance, problem.collision_tolerance,
               problem.gradient_tolerance, problem.learning_rate)
    if not all(math.isfinite(x) and x >= 0 for x in scalars) or problem.dt == 0 or problem.learning_rate == 0:
        raise ValueError("solver scalars are invalid")
    if any(not math.isfinite(x) or x < 0 for x in vars(problem.weights).values()):
        raise ValueError("weights must be finite and nonnegative")
    if problem.optimizer not in ("adam", "lbfgs", "particle", "es"):
        raise ValueError("optimizer must be 'adam', 'lbfgs', 'particle', or 'es'")
    _models(problem, start.shape[0])
    return start, goal, lower, upper, batched


def _initial_seeds(
    problem: TrajectoryProblem, start: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    if problem.seeds is None:
        return minimum_jerk_trajectory(start, goal, problem.steps)[:, None]
    seeds = _floating(problem.seeds, "seeds")
    if seeds.ndim == 3 and start.shape[0] == 1:
        seeds = seeds[None]
    if seeds.ndim != 4 or seeds.shape[0] != start.shape[0] or seeds.shape[2:] != (
        problem.steps, problem.chain.dof
    ):
        raise ValueError("seeds must have shape [N,T,J] or [B,N,T,J]")
    if seeds.device != start.device or seeds.dtype != start.dtype:
        raise ValueError("seeds must match endpoint dtype and device")
    return seeds


def optimize_trajectory(
    problem: TrajectoryProblem, *, differentiable: bool = False
) -> TrajectoryResult:
    """Deterministic projected batched Adam with hard endpoint constraints."""
    start, goal, lower, upper, batched = _validate(problem)
    seeds = _initial_seeds(problem, start, goal)
    batch, count = seeds.shape[:2]
    lo, hi = lower[:, None, None], upper[:, None, None]

    def project(value: torch.Tensor) -> torch.Tensor:
        extra = value.ndim - 4
        bounds_shape = (batch,) + (1,) * (extra + 2) + (problem.chain.dof,)
        value = torch.maximum(
            torch.minimum(value, upper.reshape(bounds_shape)),
            lower.reshape(bounds_shape),
        )
        if value.shape[1]:
            middle = value[..., 1:-1, :]
            endpoint_shape = (batch,) + (1,) * (value.ndim - 2) + (problem.chain.dof,)
            first_knot = value[..., :1, :] * 0 + start.reshape(endpoint_shape)
            last_knot = value[..., -1:, :] * 0 + goal.reshape(endpoint_shape)
            value = torch.cat((first_knot, middle, last_knot), dim=-2)
        return value

    q = project(seeds.clone())
    endpoint_feasible = (
        (start >= lower).all(-1) & (start <= upper).all(-1)
        & (goal >= lower).all(-1) & (goal <= upper).all(-1)
    )
    iterations = torch.zeros((batch, count), dtype=torch.int64, device=q.device)
    if count == 0:
        empty = q.new_empty((batch, 0))
        empty_bool = torch.empty((batch, 0), dtype=torch.bool, device=q.device)
        empty_status: tuple[tuple[str, ...], ...] = tuple(() for _ in range(batch))
        if not batched:
            return TrajectoryResult(
                q[0], empty_bool[0], (), iterations[0], empty[0], empty[0], empty[0],
                empty[0], None, False,
            )
        return TrajectoryResult(
            q, empty_bool, empty_status, iterations, empty, empty, empty, empty,
            tuple(None for _ in range(batch)), True,
        )
    stationary = torch.zeros((batch, count), dtype=torch.bool, device=q.device)
    if problem.optimizer != "adam":
        def objective(value: torch.Tensor) -> torch.Tensor:
            if value.ndim == 4:
                return evaluate_trajectory(problem, value).value
            particles = value.shape[2]
            flattened = value.reshape(batch, count * particles, problem.steps, problem.chain.dof)
            return evaluate_trajectory(problem, flattened).value.reshape(batch, count, particles)

        if problem.optimizer == "lbfgs":
            cfg = problem.lbfgs or LBFGSConfig(
                iterations=problem.max_iterations,
                learning_rate=min(1.0, problem.learning_rate * 10),
                tolerance_grad=problem.gradient_tolerance,
            )
            optimized = lbfgs_optimize(
                objective, q, config=cfg, projection=project, event_ndim=2,
                cache=problem.optimizer_cache, warm_start=problem.warm_start,
                differentiable=differentiable,
            )
        else:
            cfg = problem.particle or ParticleConfig(
                iterations=problem.max_iterations, seed=0,
            )
            optimized = particle_optimize(
                objective, q, config=cfg, projection=project, event_ndim=2,
                cache=problem.optimizer_cache,
            )
        q = optimized.solution
        stationary = optimized.converged
        iterations = optimized.iterations
        iteration = int(iterations.max().item())
    else:
        iteration = 0
        first, second = torch.zeros_like(q), torch.zeros_like(q)
        beta1, beta2 = 0.9, 0.999
        for iteration in range(1, problem.max_iterations + 1):
            q = q.requires_grad_(True)
            cost = evaluate_trajectory(problem, q)
            gradient = torch.autograd.grad(
                cost.value.sum(), q, create_graph=differentiable, retain_graph=differentiable
            )[0]
            gradient = torch.cat((torch.zeros_like(gradient[..., :1, :]),
                                  gradient[..., 1:-1, :],
                                  torch.zeros_like(gradient[..., -1:, :])), -2)
            norm = torch.linalg.vector_norm(gradient.flatten(start_dim=2), dim=-1)
            newly_stationary = (norm <= problem.gradient_tolerance) & endpoint_feasible[:, None]
            stationary |= newly_stationary
            active = endpoint_feasible[:, None] & ~stationary
            iterations = torch.where(active, torch.full_like(iterations, iteration), iterations)
            if not bool(active.any().item()):
                break
            first = beta1 * first + (1 - beta1) * gradient
            second = beta2 * second + (1 - beta2) * gradient.square()
            step = (first / (1 - beta1**iteration)) / (
                (second / (1 - beta2**iteration)).sqrt() + torch.finfo(q.dtype).eps
            )
            candidate = project(q - problem.learning_rate * step)
            q = torch.where(active[..., None, None], candidate, q)
            if not differentiable:
                q = q.detach()

    final = evaluate_trajectory(problem, q)
    endpoint_error = torch.maximum(
        torch.linalg.vector_norm(q[..., 0, :] - start[:, None], dim=-1),
        torch.linalg.vector_norm(q[..., -1, :] - goal[:, None], dim=-1),
    )
    violation = torch.maximum(
        (lower[:, None, None] - q).clamp_min(0).amax((-2, -1)),
        (q - upper[:, None, None]).clamp_min(0).amax((-2, -1)),
    )
    success = (
        endpoint_feasible[:, None]
        & (endpoint_error <= problem.endpoint_tolerance)
        & (final.minimum_clearance >= -problem.collision_tolerance)
        & (violation == 0)
    )
    statuses: list[list[str]] = []
    selected: list[int | None] = []
    for b in range(batch):
        row = []
        for n in range(count):
            if not bool(endpoint_feasible[b].item()):
                row.append("endpoint_infeasible")
            elif bool(success[b, n].item()):
                row.append("success")
            elif bool((final.minimum_clearance[b, n] < -problem.collision_tolerance).item()):
                row.append("collision_constrained")
            elif bool(stationary[b, n].item()):
                row.append("stationary")
            else:
                row.append("max_iterations")
        statuses.append(row)
        valid = torch.nonzero(success[b], as_tuple=False).flatten()
        selected.append(None if valid.numel() == 0 else int(
            valid[torch.argmin(final.value[b, valid])].item()
        ))
    if not batched:
        return TrajectoryResult(
            q[0], success[0], tuple(statuses[0]), iterations[0], final.value[0],
            endpoint_error[0], final.minimum_clearance[0], violation[0], selected[0], False,
        )
    return TrajectoryResult(
        q, success, tuple(tuple(row) for row in statuses), iterations, final.value,
        endpoint_error, final.minimum_clearance, violation, tuple(selected), True,
    )


def interpolate_trajectory(q: torch.Tensor, source_dt: float, target_dt: float) -> torch.Tensor:
    """Linearly resample a trajectory while preserving its physical duration."""
    if source_dt <= 0 or target_dt <= 0:
        raise ValueError("time steps must be positive")
    duration = (q.shape[-2] - 1) * source_dt
    steps = max(2, math.ceil(duration / target_dt) + 1)
    coordinates = torch.linspace(0, q.shape[-2] - 1, steps, device=q.device, dtype=q.dtype)
    low = coordinates.floor().to(torch.int64).clamp_max(q.shape[-2] - 2)
    fraction = coordinates - low
    fraction[-1] = 1
    shape = (1,) * (q.ndim - 2) + (-1, 1)
    return (1 - fraction).reshape(shape) * q[..., low, :] + fraction.reshape(shape) * q[..., low + 1, :]


def retime_trajectory(
    q: torch.Tensor,
    dt: float,
    *,
    velocity_limit: torch.Tensor | None = None,
    acceleration_limit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Uniformly slow a trajectory to meet per-joint velocity/acceleration limits."""
    scale = 1.0
    if velocity_limit is not None and q.shape[-2] > 1:
        ratio = torch.diff(q, dim=-2).abs() / (dt * velocity_limit)
        scale = max(scale, float(ratio.amax().item()))
    if acceleration_limit is not None and q.shape[-2] > 2:
        ratio = torch.diff(q, n=2, dim=-2).abs() / (dt * dt * acceleration_limit)
        scale = max(scale, math.sqrt(float(ratio.amax().item())))
    return q, dt * scale


def trajectory_metrics(
    problem: TrajectoryProblem, q: torch.Tensor, *, dt: float | None = None
) -> TrajectoryMetrics:
    step = problem.dt if dt is None else dt
    values = q if q.ndim >= 3 else q[None]
    velocity = torch.diff(values, dim=-2) / step
    acceleration = torch.diff(values, n=2, dim=-2) / step**2
    jerk = torch.diff(values, n=3, dim=-2) / step**3
    lower, upper = problem.lower, problem.upper
    limit = torch.maximum((lower - q).clamp_min(0), (q - upper).clamp_min(0)).amax((-2, -1))
    collision = evaluate_trajectory(problem, q if q.ndim >= 3 else q[None]).minimum_clearance
    return TrajectoryMetrics(
        (q.shape[-2] - 1) * step,
        torch.linalg.vector_norm(torch.diff(q, dim=-2), dim=-1).sum(-1),
        velocity.abs().amax((-2, -1)),
        acceleration.abs().amax((-2, -1)) if acceleration.numel() else q.sum((-2, -1)) * 0,
        jerk.abs().amax((-2, -1)) if jerk.numel() else q.sum((-2, -1)) * 0,
        collision, limit,
    )
