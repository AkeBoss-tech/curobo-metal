"""Portable seed-batched differentiable inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from curobo_metal.optim import (
    ExecutionCache, LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize,
)
from curobo_metal.ops.costs import (
    CollisionModel,
    joint_limit_cost,
    pose_cost,
    pose_error,
    robot_collision_cost,
)
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics


IKStatus = str


@dataclass(frozen=True)
class IKProblem:
    """A goal batch and shared seed batch resident on one device.

    One-dimensional pose inputs denote one goal. Solutions preserve the seed
    dimension and omit the goal dimension only for that convenience form.
    """

    chain: KinematicChain
    target_position: torch.Tensor
    target_quaternion: torch.Tensor
    seeds: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    pose_weights: torch.Tensor
    position_tolerance: float = 1e-4
    rotation_tolerance: float = 1e-4
    max_iterations: int = 300
    step_tolerance: float = 1e-8
    learning_rate: float = 0.08
    collision_model: CollisionModel | None = None
    collision_tolerance: float = 0.0
    wrap_revolute: bool = False
    optimizer: str = "adam"
    lbfgs: LBFGSConfig | None = None
    particle: ParticleConfig | None = None
    optimizer_cache: ExecutionCache | None = None
    warm_start: bool = False


@dataclass(frozen=True)
class IKResult:
    solutions: torch.Tensor
    success: torch.Tensor
    status: tuple[IKStatus, ...] | tuple[tuple[IKStatus, ...], ...]
    iterations: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor
    objective: torch.Tensor
    collision_cost: torch.Tensor
    collision_free: torch.Tensor
    selected_seed: int | None | tuple[int | None, ...]
    input_goals_were_batched: bool
    input_seeds_were_batched: bool


def _validate(problem: IKProblem) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    if not isinstance(problem.chain, KinematicChain):
        raise TypeError("chain must be a compiled KinematicChain")
    tensors = (
        problem.target_position,
        problem.target_quaternion,
        problem.seeds,
        problem.lower,
        problem.upper,
        problem.pose_weights,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("IK numeric inputs must be torch.Tensor values")
    reference = problem.seeds
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("IK tensors must be float32 or float64")
    if (
        reference.device.type != problem.chain.device.type
        or reference.dtype != problem.chain.dtype
    ):
        raise ValueError("seeds must match the chain dtype and device")
    if reference.device.type == "mps" and reference.dtype != torch.float32:
        raise TypeError("MPS IK supports only float32")
    for value in tensors:
        if value.device.type != reference.device.type or value.dtype != reference.dtype:
            raise ValueError("all IK tensors must have the same dtype and device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("IK tensors must contain only finite values")

    goals_batched = problem.target_position.ndim == 2
    position = problem.target_position
    quaternion = problem.target_quaternion
    if position.ndim == 1:
        position = position.unsqueeze(0)
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("target_position must have shape [3] or [G,3]")
    if quaternion.shape != (position.shape[0], 4):
        raise ValueError("target_quaternion must have shape [4] or [G,4]")
    if bool((torch.linalg.vector_norm(quaternion, dim=1) <= 0).any().item()):
        raise ValueError("target quaternions must be nonzero")

    seeds_batched = reference.ndim == 2
    seeds = reference.unsqueeze(0) if reference.ndim == 1 else reference
    if seeds.ndim != 2 or seeds.shape[1] != problem.chain.dof:
        raise ValueError(f"seeds must have shape [{problem.chain.dof}] or [N,{problem.chain.dof}]")
    if problem.lower.shape != (problem.chain.dof,) or problem.upper.shape != (problem.chain.dof,):
        raise ValueError("joint limits must have shape [J]")
    if bool((problem.lower > problem.upper).any().item()):
        raise ValueError("joint limits must satisfy lower <= upper")
    if problem.pose_weights.shape not in {(6,), (position.shape[0], 6)}:
        raise ValueError("pose_weights must have shape [6] or [G,6]")
    if bool((problem.pose_weights < 0).any().item()):
        raise ValueError("pose_weights must be nonnegative")
    scalars = (
        problem.position_tolerance,
        problem.rotation_tolerance,
        problem.step_tolerance,
        problem.collision_tolerance,
        problem.learning_rate,
    )
    if problem.max_iterations <= 0 or not all(math.isfinite(x) and x >= 0 for x in scalars):
        raise ValueError("iterations and solver scalars must be finite and nonnegative")
    if problem.learning_rate == 0:
        raise ValueError("learning_rate must be positive")
    if problem.optimizer not in ("adam", "lbfgs", "particle", "es"):
        raise ValueError("optimizer must be 'adam', 'lbfgs', 'particle', or 'es'")
    return position, quaternion, seeds, goals_batched, seeds_batched


def _project(problem: IKProblem, q: torch.Tensor) -> torch.Tensor:
    value = q
    if problem.wrap_revolute:
        columns = [
            index
            for kind, index in zip(problem.chain.kinds, problem.chain.q_indices, strict=True)
            if kind == "revolute" and index is not None
        ]
        if columns:
            wrapped = torch.remainder(value[..., columns] + torch.pi, 2 * torch.pi) - torch.pi
            value = value.clone()
            value[..., columns] = wrapped
    return torch.maximum(torch.minimum(value, problem.upper), problem.lower)


def _evaluate(
    problem: IKProblem,
    q: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    goals, seeds, dof = q.shape
    flat = q.reshape(goals * seeds, dof)
    all_transforms = forward_kinematics(problem.chain, flat).transforms
    transforms = all_transforms[:, -1]
    target_position = positions[:, None].expand(goals, seeds, 3).reshape(-1, 3)
    target_quaternion = quaternions[:, None].expand(goals, seeds, 4).reshape(-1, 4)
    residual = pose_error(transforms, target_position, target_quaternion).reshape(goals, seeds, 6)
    weights = problem.pose_weights
    if weights.ndim == 1:
        weights = weights.expand(goals, 6)
    objective = pose_cost(residual, weights[:, None])
    objective = objective + joint_limit_cost(q, problem.lower, problem.upper, weight=1e3)
    collision = q.new_zeros((goals, seeds))
    if problem.collision_model is not None:
        collision_flat, _ = robot_collision_cost(
            all_transforms,
            problem.collision_model,
        )
        collision = collision_flat.reshape(goals, seeds)
        objective = objective + collision
    return objective, residual, collision


def solve_ik(problem: IKProblem, *, differentiable: bool = False) -> IKResult:
    """Solve all goal/seed pairs with deterministic projected Adam.

    ``differentiable=True`` retains the unrolled optimizer graph, allowing
    derivatives from returned solutions to tensor inputs. It is more expensive
    and intended for compact training-time problems.
    """
    positions, quaternions, seeds, goals_batched, seeds_batched = _validate(problem)
    goals, count = positions.shape[0], seeds.shape[0]
    q = _project(problem, seeds[None].expand(goals, count, -1).clone())
    if count == 0:
        empty_float = q.new_empty((goals, 0))
        empty_bool = torch.empty((goals, 0), dtype=torch.bool, device=q.device)
        return _format_result(
            q, empty_bool, [], torch.empty_like(empty_bool, dtype=torch.int64),
            empty_float, empty_float, empty_float, empty_float,
            problem.collision_tolerance, goals_batched, seeds_batched,
        )

    success = torch.zeros((goals, count), dtype=torch.bool, device=q.device)
    iterations = torch.zeros((goals, count), dtype=torch.int64, device=q.device)
    stationary = torch.zeros_like(success)
    if problem.optimizer != "adam":
        def objective(value: torch.Tensor) -> torch.Tensor:
            if value.ndim == 3:
                return _evaluate(problem, value, positions, quaternions)[0]
            particles = value.shape[2]
            flat_value = value.reshape(goals, count * particles, value.shape[-1])
            return _evaluate(problem, flat_value, positions, quaternions)[0].reshape(
                goals, count, particles
            )

        if problem.optimizer == "lbfgs":
            cfg = problem.lbfgs or LBFGSConfig(
                iterations=problem.max_iterations,
                learning_rate=min(1.0, problem.learning_rate * 10),
                tolerance_grad=problem.step_tolerance,
            )
            optimized = lbfgs_optimize(
                objective, q, config=cfg, projection=lambda x: _project(problem, x),
                event_ndim=1, cache=problem.optimizer_cache,
                warm_start=problem.warm_start, differentiable=differentiable,
            )
        else:
            cfg = problem.particle or ParticleConfig(
                iterations=problem.max_iterations, seed=0,
            )
            optimized = particle_optimize(
                objective, q, config=cfg, projection=lambda x: _project(problem, x),
                event_ndim=1, cache=problem.optimizer_cache,
            )
        q = optimized.solution
        iterations = optimized.iterations
        stationary = optimized.converged
        iteration = int(iterations.max().item())
    else:
        iteration = 0
        beta1, beta2 = 0.9, 0.999
        first = torch.zeros_like(q)
        second = torch.zeros_like(q)

        for iteration in range(1, problem.max_iterations + 1):
            q = q.requires_grad_(True)
            objective_value, residual, collision = _evaluate(problem, q, positions, quaternions)
            current_success = (
                (torch.linalg.vector_norm(residual[..., :3], dim=-1) <= problem.position_tolerance)
                & (torch.linalg.vector_norm(residual[..., 3:], dim=-1) <= problem.rotation_tolerance)
                & (collision <= problem.collision_tolerance)
            )
            newly_done = (~success) & current_success
            iterations = torch.where(newly_done, torch.full_like(iterations, iteration), iterations)
            success = success | current_success
            if bool(success.all().item()):
                break
            gradient = torch.autograd.grad(
                objective_value.sum(), q, create_graph=differentiable, retain_graph=differentiable
            )[0]
            active = ~success
            gradient_norm = torch.linalg.vector_norm(gradient, dim=-1)
            became_stationary = active & (gradient_norm <= problem.step_tolerance)
            stationary = stationary | became_stationary
            active = active & ~stationary
            first = beta1 * first + (1 - beta1) * gradient
            second = beta2 * second + (1 - beta2) * gradient.square()
            corrected_first = first / (1 - beta1**iteration)
            corrected_second = second / (1 - beta2**iteration)
            candidate = _project(
                problem,
                q - problem.learning_rate * corrected_first / (
                    corrected_second.sqrt() + torch.finfo(q.dtype).eps
                ),
            )
            q = torch.where(active[..., None], candidate, q)
            if not differentiable:
                q = q.detach()
            if not bool(active.any().item()):
                break

    final_objective, final_residual, final_collision = _evaluate(
        problem, q, positions, quaternions
    )
    position_error = torch.linalg.vector_norm(final_residual[..., :3], dim=-1)
    rotation_error = torch.linalg.vector_norm(final_residual[..., 3:], dim=-1)
    final_success = (
        (position_error <= problem.position_tolerance)
        & (rotation_error <= problem.rotation_tolerance)
        & (final_collision <= problem.collision_tolerance)
    )
    newly_done = (iterations == 0) & final_success
    iterations = torch.where(newly_done, torch.full_like(iterations, iteration), iterations)
    iterations = torch.where(iterations == 0, torch.full_like(iterations, iteration), iterations)
    success = final_success

    at_limit = torch.isclose(q, problem.lower, atol=1e-6, rtol=0).any(-1) | torch.isclose(
        q, problem.upper, atol=1e-6, rtol=0
    ).any(-1)
    statuses: list[list[str]] = []
    for goal in range(goals):
        row: list[str] = []
        for seed in range(count):
            if bool(success[goal, seed].item()):
                row.append("success")
            elif bool((final_collision[goal, seed] > problem.collision_tolerance).item()):
                row.append("collision_constrained")
            elif bool(at_limit[goal, seed].item()):
                row.append("limit_constrained")
            elif bool(stationary[goal, seed].item()):
                row.append("infeasible_or_stationary")
            else:
                row.append("max_iterations")
        statuses.append(row)
    return _format_result(
        q, success, statuses, iterations, position_error, rotation_error,
        final_objective, final_collision, problem.collision_tolerance,
        goals_batched, seeds_batched,
    )


def _format_result(
    q: torch.Tensor,
    success: torch.Tensor,
    statuses: list[list[str]],
    iterations: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    objective: torch.Tensor,
    collision: torch.Tensor,
    collision_tolerance: float,
    goals_batched: bool,
    seeds_batched: bool,
) -> IKResult:
    selected: list[int | None] = []
    for goal in range(q.shape[0]):
        indices = torch.nonzero(success[goal], as_tuple=False).flatten()
        selected.append(
            None if indices.numel() == 0
            else int(indices[torch.argmin(objective[goal, indices])].item())
        )
    collision_free = collision <= collision_tolerance
    if not goals_batched:
        return IKResult(
            q[0], success[0], tuple(statuses[0]) if statuses else tuple(),
            iterations[0], position[0], rotation[0], objective[0], collision[0],
            collision_free[0], selected[0], False, seeds_batched,
        )
    return IKResult(
        q, success, tuple(tuple(row) for row in statuses), iterations, position,
        rotation, objective, collision, collision_free, tuple(selected), True, seeds_batched,
    )
