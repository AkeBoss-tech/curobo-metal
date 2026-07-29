"""Dynamics-aware B-spline trajectory optimization.

The implementation is intentionally composed from ordinary PyTorch operators so
the same differentiable code path runs on CPU and MPS without fallback kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory.core import trajectory_collision_cost
from curobo_metal.ops.whole_body import WholeBodyModel, inverse_dynamics

DynamicsAwareStatus = str


@dataclass(frozen=True)
class BSplineMatrices:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    parameters: torch.Tensor


@dataclass(frozen=True)
class DynamicsAwareWeights:
    smoothness: float = 1.0
    effort: float = 1e-3
    duration: float = 1e-2
    joint_limit: float = 100.0
    velocity_limit: float = 100.0
    acceleration_limit: float = 100.0
    jerk_limit: float = 10.0
    torque_limit: float = 100.0
    collision: float = 1.0


@dataclass(frozen=True)
class DynamicsAwareProblem:
    model: WholeBodyModel
    start: torch.Tensor
    goal: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    control_points: int = 10
    samples: int = 32
    degree: int = 3
    duration: float = 2.0
    min_duration: float = 0.1
    max_duration: float = 10.0
    optimize_timestep: bool = True
    velocity_limits: torch.Tensor | None = None
    acceleration_limits: torch.Tensor | None = None
    jerk_limits: torch.Tensor | None = None
    torque_limits: torch.Tensor | None = None
    seeds: torch.Tensor | None = None
    weights: DynamicsAwareWeights = DynamicsAwareWeights()
    chain: KinematicChain | None = None
    collision_model: CollisionModel | Sequence[CollisionModel | None] | None = None
    collision_subdivisions: int = 1
    max_iterations: int = 150
    learning_rate: float = 0.03
    gradient_tolerance: float = 1e-7
    constraint_tolerance: float = 1e-4


@dataclass(frozen=True)
class DynamicsAwareCost:
    total: torch.Tensor
    smoothness: torch.Tensor
    effort: torch.Tensor
    duration: torch.Tensor
    joint_limit: torch.Tensor
    velocity_limit: torch.Tensor
    acceleration_limit: torch.Tensor
    jerk_limit: torch.Tensor
    torque_limit: torch.Tensor
    collision: torch.Tensor
    minimum_clearance: torch.Tensor
    maximum_violation: torch.Tensor
    torque: torch.Tensor


@dataclass(frozen=True)
class DynamicsAwareResult:
    control_points: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    torque: torch.Tensor
    duration: torch.Tensor
    success: torch.Tensor
    status: tuple[DynamicsAwareStatus, ...] | tuple[tuple[DynamicsAwareStatus, ...], ...]
    iterations: torch.Tensor
    objective: torch.Tensor
    maximum_violation: torch.Tensor
    minimum_clearance: torch.Tensor
    selected_seed: int | None | tuple[int | None, ...]
    input_problems_were_batched: bool


def _basis_degree(
    parameters: torch.Tensor, knots: torch.Tensor, degree: int, count: int
) -> list[torch.Tensor]:
    # The right endpoint belongs to the final span for a clamped spline.
    rows = [
        ((parameters >= knots[i]) & (parameters < knots[i + 1])).to(parameters.dtype)
        for i in range(count + degree)
    ]
    rows[count - 1] = torch.where(
        parameters == knots[-1], torch.ones_like(rows[count - 1]), rows[count - 1]
    )
    for current in range(1, degree + 1):
        next_rows: list[torch.Tensor] = []
        for i in range(count + degree - current):
            left_den = float((knots[i + current] - knots[i]).item())
            right_den = float((knots[i + current + 1] - knots[i + 1]).item())
            left = 0.0 if left_den == 0 else (parameters - knots[i]) / left_den * rows[i]
            right = (
                0.0
                if right_den == 0
                else (knots[i + current + 1] - parameters) / right_den * rows[i + 1]
            )
            next_rows.append(left + right)
        rows = next_rows
    return rows[:count]


def _derivative_coefficients(
    count: int, degree: int, knots: torch.Tensor, order: int
) -> torch.Tensor:
    matrix = torch.eye(count, dtype=knots.dtype, device=knots.device)
    current_count, current_degree = count, degree
    for _ in range(order):
        if current_degree <= 0:
            return knots.new_zeros((0, count))
        difference = knots.new_zeros((current_count - 1, current_count))
        for i in range(current_count - 1):
            denominator = knots[i + current_degree + 1] - knots[i + 1]
            coefficient = current_degree / denominator
            difference[i, i] = -coefficient
            difference[i, i + 1] = coefficient
        matrix = difference @ matrix
        knots = knots[1:-1]
        current_count -= 1
        current_degree -= 1
    return matrix


def bspline_matrices(
    control_points: int,
    samples: int,
    *,
    degree: int = 3,
    duration: float | torch.Tensor = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BSplineMatrices:
    """Return clamped-uniform B-spline sampling and physical derivative matrices."""
    if isinstance(control_points, bool) or control_points < 2:
        raise ValueError("control_points must be an integer >= 2")
    if isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if isinstance(degree, bool) or not 1 <= degree < control_points:
        raise ValueError("degree must satisfy 1 <= degree < control_points")
    parameters = torch.linspace(0, 1, samples, device=device, dtype=dtype)
    interior_count = control_points - degree - 1
    interior = (
        torch.arange(1, interior_count + 1, device=device, dtype=dtype)
        / (interior_count + 1)
    )
    knots = torch.cat((parameters.new_zeros(degree + 1), interior, parameters.new_ones(degree + 1)))
    matrices: list[torch.Tensor] = []
    for order in range(4):
        if order > degree:
            matrices.append(parameters.new_zeros((samples, control_points)))
            continue
        derivative_knots = knots[order : knots.numel() - order]
        derivative_basis = torch.stack(
            _basis_degree(
                parameters, derivative_knots, degree - order, control_points - order
            ),
            dim=-1,
        )
        coefficients = _derivative_coefficients(control_points, degree, knots, order)
        matrices.append(derivative_basis @ coefficients)
    duration_tensor = torch.as_tensor(duration, device=device, dtype=dtype)
    if duration_tensor.ndim != 0 or not bool(torch.isfinite(duration_tensor).item()) or duration_tensor <= 0:
        raise ValueError("duration must be a positive finite scalar")
    return BSplineMatrices(
        matrices[0],
        matrices[1] / duration_tensor,
        matrices[2] / duration_tensor.square(),
        matrices[3] / duration_tensor.pow(3),
        parameters,
    )


def sample_bspline(
    control_points: torch.Tensor, matrices: BSplineMatrices
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample position through jerk, preserving all control-point leading axes."""
    if control_points.ndim < 2 or control_points.shape[-2] != matrices.position.shape[-1]:
        raise ValueError("control_points must end in [C,J] matching the matrices")
    return tuple(
        torch.einsum("sc,...cj->...sj", matrix, control_points)
        for matrix in (
            matrices.position,
            matrices.velocity,
            matrices.acceleration,
            matrices.jerk,
        )
    )  # type: ignore[return-value]


def _limit_violation(value: torch.Tensor, limit: torch.Tensor | None) -> torch.Tensor:
    if limit is None:
        return value.new_zeros(value.shape)
    return (value.abs() - limit).clamp_min(0)


def _sum_square(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum(dim=(-2, -1))


def evaluate_dynamics_aware(
    problem: DynamicsAwareProblem,
    control_points: torch.Tensor,
    duration: torch.Tensor | float,
) -> DynamicsAwareCost:
    """Evaluate a normalized ``[B,N,C,J]`` population."""
    cp = control_points
    if cp.ndim == 3:
        cp = cp.unsqueeze(0)
    if cp.ndim != 4:
        raise ValueError("control_points must have shape [N,C,J] or [B,N,C,J]")
    duration_t = torch.as_tensor(duration, device=cp.device, dtype=cp.dtype)
    if duration_t.ndim == 0:
        duration_t = duration_t.expand(cp.shape[:2])
    elif duration_t.ndim == 1 and cp.shape[0] == 1:
        duration_t = duration_t.unsqueeze(0)
    if duration_t.shape != cp.shape[:2]:
        raise ValueError("duration must be scalar or match [B,N]")
    unit = bspline_matrices(
        cp.shape[-2], problem.samples, degree=problem.degree,
        device=cp.device, dtype=cp.dtype,
    )
    q = torch.einsum("sc,bncj->bnsj", unit.position, cp)
    qd_u = torch.einsum("sc,bncj->bnsj", unit.velocity, cp)
    qdd_u = torch.einsum("sc,bncj->bnsj", unit.acceleration, cp)
    qddd_u = torch.einsum("sc,bncj->bnsj", unit.jerk, cp)
    scale = duration_t[..., None, None]
    qd, qdd, qddd = qd_u / scale, qdd_u / scale.square(), qddd_u / scale.pow(3)
    flat = lambda x: x.reshape(-1, x.shape[-1])
    torque = inverse_dynamics(problem.model, flat(q), flat(qd), flat(qdd)).torque.reshape_as(q)

    lower = problem.lower
    upper = problem.upper
    if lower.ndim == 1:
        lower, upper = lower[None], upper[None]
    joint_v = (lower[:, None, None] - q).clamp_min(0) + (q - upper[:, None, None]).clamp_min(0)
    def expand_limit(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if value.ndim == 1:
            return value[None, None, None]
        if value.ndim == 2 and value.shape[0] == q.shape[0]:
            return value[:, None, None]
        raise ValueError("dynamic limits must have shape [J] or [B,J]")
    velocity_v = _limit_violation(qd, expand_limit(problem.velocity_limits))
    acceleration_v = _limit_violation(qdd, expand_limit(problem.acceleration_limits))
    jerk_v = _limit_violation(qddd, expand_limit(problem.jerk_limits))
    torque_limits = problem.torque_limits
    if torque_limits is None:
        torque_limits = problem.model.effort_limits
    torque_v = _limit_violation(torque, expand_limit(torque_limits))
    collision = q.new_zeros(q.shape[:2])
    clearance = q.new_full(q.shape[:2], torch.inf)
    if problem.chain is not None:
        models = problem.collision_model
        if isinstance(models, Sequence):
            if len(models) != q.shape[0]:
                raise ValueError("collision model sequence must match batch")
            model_rows = tuple(models)
        else:
            model_rows = (models,) * q.shape[0]
        for b, model in enumerate(model_rows):
            collision[b], clearance[b] = trajectory_collision_cost(
                problem.chain, q[b], model, subdivisions=problem.collision_subdivisions
            )
    w = problem.weights
    dt = duration_t / (problem.samples - 1)
    smooth = 0.5 * w.smoothness * _sum_square(qddd) * dt
    effort = 0.5 * w.effort * _sum_square(torque) * dt
    duration_cost = w.duration * duration_t
    terms = (
        0.5 * w.joint_limit * _sum_square(joint_v),
        0.5 * w.velocity_limit * _sum_square(velocity_v),
        0.5 * w.acceleration_limit * _sum_square(acceleration_v),
        0.5 * w.jerk_limit * _sum_square(jerk_v),
        0.5 * w.torque_limit * _sum_square(torque_v),
    )
    collision_cost = w.collision * collision
    maxima = torch.stack(
        [
            value.amax(dim=(-2, -1))
            for value in (joint_v, velocity_v, acceleration_v, jerk_v, torque_v)
        ],
        dim=-1,
    ).amax(-1)
    total = smooth + effort + duration_cost + sum(terms) + collision_cost
    return DynamicsAwareCost(
        total, smooth, effort, duration_cost, *terms, collision_cost, clearance,
        maxima, torque,
    )


def _validate(problem: DynamicsAwareProblem) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if not isinstance(problem.model, WholeBodyModel):
        raise TypeError("model must be a WholeBodyModel")
    start, goal = problem.start, problem.goal
    if start.ndim == 1:
        start, goal, batched = start[None], goal[None], False
    else:
        batched = True
    if start.ndim != 2 or goal.shape != start.shape or start.shape[-1] != problem.model.dof:
        raise ValueError("start and goal must have equal shape [J] or [B,J]")
    for value in (start, goal, problem.lower, problem.upper):
        if value.device.type != problem.model.device.type or value.dtype != problem.model.dtype:
            raise ValueError("problem tensors must match the dynamics model")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("problem tensors must be finite")
    if problem.control_points <= problem.degree or problem.samples < 2:
        raise ValueError("invalid B-spline dimensions")
    if not (0 < problem.min_duration <= problem.duration <= problem.max_duration):
        raise ValueError("duration bounds must contain duration and be positive")
    if problem.chain is None and problem.collision_model is not None:
        raise ValueError("chain is required when collision_model is provided")
    if problem.chain is not None and (
        problem.chain.dof != problem.model.dof
        or problem.chain.device != problem.model.device
        or problem.chain.dtype != problem.model.dtype
    ):
        raise ValueError("collision chain must match dynamics model DOF, device, and dtype")
    return start, goal, batched


def _initial_control_points(problem: DynamicsAwareProblem, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    phase = torch.linspace(
        0, 1, problem.control_points, device=start.device, dtype=start.dtype
    )
    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    default = start[:, None, :] + blend[None, :, None] * (goal - start)[:, None, :]
    if problem.seeds is None:
        return default[:, None]
    seeds = problem.seeds
    if seeds.ndim == 3:
        if start.shape[0] != 1:
            raise ValueError("batched problems require seeds shaped [B,N,C,J]")
        seeds = seeds[None]
    if seeds.ndim != 4 or seeds.shape[0] != start.shape[0] or seeds.shape[-2:] != default.shape[-2:]:
        raise ValueError("seeds must have shape [N,C,J] or [B,N,C,J]")
    return seeds


def optimize_dynamics_aware(problem: DynamicsAwareProblem) -> DynamicsAwareResult:
    """Optimize batched B-spline seeds with deterministic feasibility statuses."""
    start, goal, batched = _validate(problem)
    cp = _initial_control_points(problem, start, goal).clone()
    batch, seeds = cp.shape[:2]
    lower, upper = problem.lower, problem.upper
    if lower.ndim == 1:
        lower, upper = lower[None], upper[None]
    lo, hi = lower[:, None, None], upper[:, None, None]

    def project(value: torch.Tensor) -> torch.Tensor:
        value = torch.maximum(torch.minimum(value, hi), lo)
        return torch.cat(
            (
                start[:, None, None].expand(-1, seeds, 1, -1),
                value[..., 1:-1, :],
                goal[:, None, None].expand(-1, seeds, 1, -1),
            ),
            dim=-2,
        )

    cp = project(cp)
    log_duration = cp.new_full((batch, seeds), math.log(problem.duration))
    first_cp, second_cp = torch.zeros_like(cp), torch.zeros_like(cp)
    first_t, second_t = torch.zeros_like(log_duration), torch.zeros_like(log_duration)
    stationary = torch.zeros((batch, seeds), dtype=torch.bool, device=cp.device)
    iterations = torch.zeros((batch, seeds), dtype=torch.int64, device=cp.device)
    beta1, beta2 = 0.9, 0.999
    iteration = 0
    for iteration in range(1, problem.max_iterations + 1):
        cp = cp.requires_grad_(True)
        log_duration = log_duration.requires_grad_(problem.optimize_timestep)
        duration = log_duration.exp()
        cost = evaluate_dynamics_aware(problem, cp, duration)
        variables = (cp, log_duration) if problem.optimize_timestep else (cp,)
        gradients = torch.autograd.grad(cost.total.sum(), variables)
        grad_cp = gradients[0]
        grad_t = gradients[1] if problem.optimize_timestep else torch.zeros_like(log_duration)
        norms = torch.sqrt(
            grad_cp.square().sum((-2, -1)) + grad_t.square()
        )
        newly_stationary = norms <= problem.gradient_tolerance
        active = ~stationary
        iterations[active] = iteration
        stationary = stationary | newly_stationary
        mask_cp = active[..., None, None].to(cp.dtype)
        mask_t = active.to(cp.dtype)
        first_cp = beta1 * first_cp + (1 - beta1) * grad_cp
        second_cp = beta2 * second_cp + (1 - beta2) * grad_cp.square()
        first_t = beta1 * first_t + (1 - beta1) * grad_t
        second_t = beta2 * second_t + (1 - beta2) * grad_t.square()
        correction1, correction2 = 1 - beta1**iteration, 1 - beta2**iteration
        with torch.no_grad():
            cp = project(
                cp - problem.learning_rate * mask_cp
                * (first_cp / correction1) / ((second_cp / correction2).sqrt() + 1e-8)
            )
            log_duration = (
                log_duration - problem.learning_rate * mask_t
                * (first_t / correction1) / ((second_t / correction2).sqrt() + 1e-8)
            ).clamp(math.log(problem.min_duration), math.log(problem.max_duration))
        if bool(stationary.all().item()):
            break

    duration = log_duration.exp().detach()
    cp = cp.detach()
    final = evaluate_dynamics_aware(problem, cp, duration)
    unit = bspline_matrices(
        problem.control_points, problem.samples, degree=problem.degree,
        device=cp.device, dtype=cp.dtype,
    )
    q = torch.einsum("sc,bncj->bnsj", unit.position, cp)
    qd = torch.einsum("sc,bncj->bnsj", unit.velocity, cp) / duration[..., None, None]
    qdd = torch.einsum("sc,bncj->bnsj", unit.acceleration, cp) / duration[..., None, None].square()
    jerk = torch.einsum("sc,bncj->bnsj", unit.jerk, cp) / duration[..., None, None].pow(3)
    collision_ok = final.minimum_clearance >= 0
    success = (final.maximum_violation <= problem.constraint_tolerance) & collision_ok
    finite = torch.isfinite(final.total)
    endpoint_ok = (
        ((start >= lower) & (start <= upper) & (goal >= lower) & (goal <= upper)).all(-1)
    )
    status_rows: list[tuple[str, ...]] = []
    selected: list[int | None] = []
    for b in range(batch):
        row = []
        for n in range(seeds):
            if not bool(finite[b, n].item()):
                row.append("NUMERICAL_FAILURE")
            elif not bool(endpoint_ok[b].item()):
                row.append("INVALID_ENDPOINT")
            elif bool(success[b, n].item()):
                row.append("SUCCESS")
            elif not bool(collision_ok[b, n].item()):
                row.append("COLLISION")
            else:
                row.append("CONSTRAINT_VIOLATION")
        status_rows.append(tuple(row))
        feasible = torch.nonzero(success[b], as_tuple=False).flatten()
        if feasible.numel():
            values = final.total[b, feasible]
            selected.append(int(feasible[torch.argmin(values)].item()))
        else:
            selected.append(None)
    if not batched:
        return DynamicsAwareResult(
            cp[0], q[0], qd[0], qdd[0], jerk[0], final.torque[0], duration[0],
            success[0], status_rows[0], iterations[0], final.total[0],
            final.maximum_violation[0], final.minimum_clearance[0], selected[0], False,
        )
    return DynamicsAwareResult(
        cp, q, qd, qdd, jerk, final.torque, duration, success,
        tuple(status_rows), iterations, final.total, final.maximum_violation,
        final.minimum_clearance, tuple(selected), True,
    )


def retime_dynamics_aware(
    problem: DynamicsAwareProblem,
    control_points: torch.Tensor,
    *,
    iterations: int = 48,
) -> torch.Tensor:
    """Find the shortest feasible duration for fixed control points by bisection."""
    low = control_points.new_full(control_points.shape[:-2], problem.min_duration)
    high = control_points.new_full(control_points.shape[:-2], problem.max_duration)
    for _ in range(iterations):
        middle = (low + high) * 0.5
        cost = evaluate_dynamics_aware(problem, control_points, middle)
        feasible = (
            (cost.maximum_violation <= problem.constraint_tolerance)
            & (cost.minimum_clearance >= 0)
        )
        if control_points.ndim == 3:
            feasible = feasible.squeeze(0)
        high = torch.where(feasible, middle, high)
        low = torch.where(feasible, low, middle)
    return high


__all__ = [
    "BSplineMatrices",
    "DynamicsAwareCost",
    "DynamicsAwareProblem",
    "DynamicsAwareResult",
    "DynamicsAwareStatus",
    "DynamicsAwareWeights",
    "bspline_matrices",
    "evaluate_dynamics_aware",
    "optimize_dynamics_aware",
    "retime_dynamics_aware",
    "sample_bspline",
]
