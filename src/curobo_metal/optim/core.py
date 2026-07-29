"""Portable, device-resident batched optimizers.

The leading dimensions are independent solver items and the final dimensions
described by ``event_ndim`` form one optimization vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Literal

import torch

Objective = Callable[[torch.Tensor], torch.Tensor]
Projection = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ParticleConfig:
    iterations: int = 32
    particles: int = 32
    elite_count: int = 8
    initial_std: float = 0.25
    minimum_std: float = 1e-4
    momentum: float = 0.1
    seed: int = 0
    tolerance: float = 1e-7
    strategy: Literal["cem", "mppi", "random"] = "cem"
    temperature: float = 1.0
    covariance: torch.Tensor | None = field(default=None, compare=False, hash=False, repr=False)
    record_debug: bool = False

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.particles <= 0:
            raise ValueError("particle iterations and count must be positive")
        if not 0 < self.elite_count <= self.particles:
            raise ValueError("elite_count must be in [1, particles]")
        values = (self.initial_std, self.minimum_std, self.momentum, self.tolerance, self.temperature)
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("particle scalars must be finite and nonnegative")
        if (self.initial_std == 0 or not 0 <= self.momentum < 1 or self.seed < 0
                or self.temperature == 0 or self.strategy not in ("cem", "mppi", "random")):
            raise ValueError("invalid particle scale, momentum, or seed")


@dataclass(frozen=True)
class LBFGSConfig:
    iterations: int = 100
    history_size: int = 10
    learning_rate: float = 1.0
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    line_search_steps: int = 12
    line_search_decay: float = 0.5
    c1: float = 1e-4
    c2: float = 0.9
    line_search: Literal["armijo", "strong_wolfe", "none"] = "armijo"
    termination: Literal["either", "gradient", "change", "both"] = "either"
    record_debug: bool = False

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.history_size < 0 or self.line_search_steps <= 0:
            raise ValueError("iterations/line-search must be positive and history nonnegative")
        values = (
            self.learning_rate, self.tolerance_grad, self.tolerance_change,
            self.line_search_decay, self.c1, self.c2,
        )
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("L-BFGS scalars must be finite and nonnegative")
        if (self.learning_rate == 0 or not 0 < self.line_search_decay < 1
                or not 0 < self.c1 < self.c2 < 1
                or self.line_search not in ("armijo", "strong_wolfe", "none")
                or self.termination not in ("either", "gradient", "change", "both")):
            raise ValueError("invalid L-BFGS learning rate or line-search parameters")


@dataclass(frozen=True)
class OptimizerResult:
    solution: torch.Tensor
    objective: torch.Tensor
    converged: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor
    status: tuple[str, ...]
    final_gradient_norm: torch.Tensor | None = None
    debug: dict[str, tuple[torch.Tensor, ...]] | None = None


@dataclass
class ExecutionCache:
    """Reusable shape-keyed state; this intentionally does not capture CUDA graphs."""

    capacity: int | None = None
    generation: int = 0
    hits: int = 0
    misses: int = 0
    _entries: dict[tuple[object, ...], dict[str, object]] = field(default_factory=dict)

    def acquire(self, key: tuple[object, ...]) -> dict[str, object]:
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        if self.capacity == 0:
            return {}
        if self.capacity is not None and self.capacity > 0 and len(self._entries) >= self.capacity:
            self._entries.pop(next(iter(self._entries)))
        value: dict[str, object] = {}
        self._entries[key] = value
        return value

    def reset(self) -> None:
        self._entries.clear()
        self.generation += 1

    clear = reset

    def update(self, *, capacity: int | None = None) -> None:
        if capacity is not None and capacity < 0:
            raise ValueError("cache capacity must be nonnegative or None")
        self.capacity = capacity
        while capacity is not None and len(self._entries) > capacity:
            self._entries.pop(next(iter(self._entries)))
        self.generation += 1

    @property
    def size(self) -> int:
        return len(self._entries)


def _shape_key(name: str, value: torch.Tensor, event_ndim: int, config: object) -> tuple[object, ...]:
    return (name, tuple(value.shape), str(value.dtype), value.device.type, event_ndim, config)


def _flatten(value: torch.Tensor, event_ndim: int) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if event_ndim <= 0 or event_ndim > value.ndim:
        raise ValueError("event_ndim must identify one or more trailing dimensions")
    batch, event = value.shape[:-event_ndim], value.shape[-event_ndim:]
    return value.reshape((-1, math.prod(event))), batch, event


def _result(
    flat: torch.Tensor, event: tuple[int, ...], batch: tuple[int, ...], objective: torch.Tensor,
    converged: torch.Tensor, failed: torch.Tensor, iterations: torch.Tensor,
    *, final_gradient_norm: torch.Tensor | None = None,
    debug: dict[str, tuple[torch.Tensor, ...]] | None = None,
) -> OptimizerResult:
    labels = [
        "nonfinite" if bool(failed[i].item()) else
        "converged" if bool(converged[i].item()) else "max_iterations"
        for i in range(flat.shape[0])
    ]
    return OptimizerResult(
        flat.reshape(batch + event), objective.reshape(batch), converged.reshape(batch),
        failed.reshape(batch), iterations.reshape(batch), tuple(labels),
        None if final_gradient_norm is None else final_gradient_norm.reshape(batch), debug,
    )


def particle_optimize(
    objective: Objective,
    initial: torch.Tensor,
    *,
    config: ParticleConfig = ParticleConfig(),
    projection: Projection | None = None,
    event_ndim: int = 1,
    cache: ExecutionCache | None = None,
) -> OptimizerResult:
    """Cross-entropy particle evolution with a CPU-seeded portable RNG stream."""
    flat, batch_shape, event_shape = _flatten(initial, event_ndim)
    count, width = flat.shape
    if count == 0:
        empty = flat.new_empty((0,))
        return _result(flat, event_shape, batch_shape, empty, empty.bool(), empty.bool(), empty.long())
    entry = (cache or ExecutionCache(capacity=0)).acquire(
        _shape_key("particle", initial, event_ndim, config)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    mean = flat.clone()
    std = torch.full_like(mean, config.initial_std)
    best = mean
    best_value = objective(best.reshape(batch_shape + event_shape)).reshape(count)
    failed = ~torch.isfinite(best_value)
    converged = torch.zeros(count, dtype=torch.bool, device=initial.device)
    iterations = torch.zeros(count, dtype=torch.int64, device=initial.device)
    previous = best_value
    objective_history: list[torch.Tensor] = []
    scale_history: list[torch.Tensor] = []
    covariance = config.covariance
    if covariance is not None:
        covariance = covariance.to(device=initial.device, dtype=initial.dtype)
        if covariance.shape == (width,):
            covariance = torch.diag(covariance)
        if covariance.shape != (width, width):
            raise ValueError("covariance must have shape [D] or [D,D]")
        if not bool(torch.isfinite(covariance).all().item()):
            raise ValueError("covariance must be finite")
        factor = torch.linalg.cholesky(covariance)
    else:
        factor = None
    for iteration in range(1, config.iterations + 1):
        noise = torch.randn(
            (count, config.particles, width), generator=generator, dtype=initial.dtype,
            device="cpu",
        ).to(initial.device)
        if factor is not None:
            noise = noise @ factor.transpose(-1, -2)
        population = mean[:, None] + std[:, None] * noise
        population[:, 0] = best
        shaped = population.reshape(batch_shape + (config.particles,) + event_shape)
        if projection is not None:
            shaped = projection(shaped)
            population = shaped.reshape(count, config.particles, width)
        values = objective(shaped).reshape(count, config.particles)
        finite = torch.isfinite(values)
        ranked = torch.where(finite, values, torch.full_like(values, torch.inf))
        elite_index = torch.topk(ranked, config.elite_count, largest=False).indices
        elite = torch.gather(population, 1, elite_index[..., None].expand(-1, -1, width))
        elite_values = torch.gather(ranked, 1, elite_index)
        if config.strategy == "mppi":
            baseline = ranked.amin(1, keepdim=True)
            weights = torch.softmax(-(ranked - baseline) / config.temperature, dim=1)
            weights = torch.where(finite, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            next_mean = (weights[..., None] * population).sum(1)
            next_std = (
                weights[..., None] * (population - next_mean[:, None]).square()
            ).sum(1).sqrt().clamp_min(config.minimum_std)
        elif config.strategy == "random":
            next_mean, next_std = mean, std
        else:
            next_mean = elite.mean(1)
            next_std = elite.std(1, unbiased=False).clamp_min(config.minimum_std)
        mean = config.momentum * mean + (1 - config.momentum) * next_mean
        std = config.momentum * std + (1 - config.momentum) * next_std
        chosen = elite_values[:, 0] < best_value
        best = torch.where(chosen[:, None], elite[:, 0], best)
        best_value = torch.where(chosen, elite_values[:, 0], best_value)
        delta = (previous - best_value).abs()
        newly = (~failed) & ((delta <= config.tolerance) | (std.amax(1) <= config.minimum_std))
        iterations = torch.where((iterations == 0) & newly, torch.full_like(iterations, iteration), iterations)
        converged |= newly
        failed |= ~finite.any(1)
        previous = best_value
        if config.record_debug:
            objective_history.append(best_value.detach().clone())
            scale_history.append(std.detach().clone())
    iterations = torch.where(iterations == 0, torch.full_like(iterations, config.iterations), iterations)
    entry["last_solution"] = best.detach()
    debug = None
    if config.record_debug:
        debug = {"objective": tuple(objective_history), "scale": tuple(scale_history)}
    return _result(best, event_shape, batch_shape, best_value, converged, failed, iterations, debug=debug)


def lbfgs_optimize(
    objective: Objective,
    initial: torch.Tensor,
    *,
    config: LBFGSConfig = LBFGSConfig(),
    projection: Projection | None = None,
    event_ndim: int = 1,
    cache: ExecutionCache | None = None,
    warm_start: bool = False,
    differentiable: bool = False,
) -> OptimizerResult:
    """Independent batched limited-memory BFGS with Armijo backtracking."""
    flat, batch_shape, event_shape = _flatten(initial, event_ndim)
    count, width = flat.shape
    if count == 0:
        empty = flat.new_empty((0,))
        return _result(flat, event_shape, batch_shape, empty, empty.bool(), empty.bool(), empty.long())
    entry = (cache or ExecutionCache(capacity=0)).acquire(
        _shape_key("lbfgs", initial, event_ndim, config)
    )
    x = flat.clone()
    previous_solution = entry.get("last_solution")
    if warm_start and isinstance(previous_solution, torch.Tensor) and previous_solution.shape == x.shape:
        x = previous_solution.to(device=x.device, dtype=x.dtype).clone()
    s_hist: list[torch.Tensor] = []
    y_hist: list[torch.Tensor] = []
    rho_hist: list[torch.Tensor] = []
    converged = torch.zeros(count, dtype=torch.bool, device=x.device)
    failed = torch.zeros_like(converged)
    iterations = torch.zeros(count, dtype=torch.int64, device=x.device)
    objective_history: list[torch.Tensor] = []
    gradient_history: list[torch.Tensor] = []

    def value_grad(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leaf = point.requires_grad_(True)
        value = objective(leaf.reshape(batch_shape + event_shape)).reshape(count)
        safe_value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        if safe_value.requires_grad:
            gradient_value = torch.autograd.grad(
                safe_value.sum(), leaf, create_graph=differentiable,
                retain_graph=differentiable, allow_unused=True,
            )[0]
            gradient = torch.zeros_like(leaf) if gradient_value is None else gradient_value
        else:
            gradient = torch.zeros_like(leaf)
        return value, gradient

    value, gradient = value_grad(x)
    failed |= ~torch.isfinite(value) | ~torch.isfinite(gradient).all(1)
    for iteration in range(1, config.iterations + 1):
        grad_norm = torch.linalg.vector_norm(gradient, dim=1)
        gradient_done = grad_norm <= config.tolerance_grad
        newly = (~failed) & gradient_done if config.termination in ("either", "gradient") else torch.zeros_like(failed)
        converged |= newly
        iterations = torch.where((iterations == 0) & newly, torch.full_like(iterations, iteration - 1), iterations)
        active = ~(converged | failed)
        if not bool(active.any().item()):
            break
        q = gradient
        alphas: list[torch.Tensor] = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            alpha = rho * (s * q).sum(1)
            alphas.append(alpha)
            q = q - alpha[:, None] * y
        if s_hist:
            sy = (s_hist[-1] * y_hist[-1]).sum(1)
            yy = y_hist[-1].square().sum(1).clamp_min(torch.finfo(x.dtype).eps)
            q = q * (sy / yy).clamp_min(torch.finfo(x.dtype).eps)[:, None]
        for s, y, rho, alpha in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            beta = rho * (y * q).sum(1)
            q = q + (alpha - beta)[:, None] * s
        direction = -q
        descent = (gradient * direction).sum(1)
        direction = torch.where((descent < 0)[:, None], direction, -gradient)
        descent = (gradient * direction).sum(1)
        accepted = torch.zeros(count, dtype=torch.bool, device=x.device)
        candidate, candidate_value = x, value
        step = torch.full((count,), config.learning_rate, dtype=x.dtype, device=x.device)
        search_steps = 1 if config.line_search == "none" else config.line_search_steps
        for _ in range(search_steps):
            trial = x + step[:, None] * direction
            shaped = trial.reshape(batch_shape + event_shape)
            if projection is not None:
                shaped = projection(shaped)
                trial = shaped.reshape(count, width)
            trial_value = objective(shaped).reshape(count)
            armijo = torch.isfinite(trial_value) & (
                trial_value <= value + config.c1 * step * descent
            )
            if config.line_search == "none":
                armijo = torch.isfinite(trial_value)
            elif config.line_search == "strong_wolfe":
                _, trial_gradient = value_grad(trial)
                curvature_ok = (trial_gradient * direction).sum(1).abs() <= config.c2 * descent.abs()
                armijo &= curvature_ok
            take = active & ~accepted & armijo
            candidate = torch.where(take[:, None], trial, candidate)
            candidate_value = torch.where(take, trial_value, candidate_value)
            accepted |= take
            step = torch.where(accepted, step, step * config.line_search_decay)
        line_failed = active & ~accepted
        failed |= line_failed
        next_value, next_gradient = value_grad(candidate)
        s, y = candidate - x, next_gradient - gradient
        curvature = (s * y).sum(1)
        valid_history = accepted & torch.isfinite(curvature) & (
            curvature > torch.finfo(x.dtype).eps
        )
        if config.history_size and bool(valid_history.any().item()):
            safe = torch.where(valid_history, curvature, torch.ones_like(curvature))
            s_hist.append(torch.where(valid_history[:, None], s, torch.zeros_like(s)))
            y_hist.append(torch.where(valid_history[:, None], y, torch.zeros_like(y)))
            rho_hist.append(torch.where(valid_history, safe.reciprocal(), torch.zeros_like(safe)))
            if len(s_hist) > config.history_size:
                s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)
        change = (candidate_value - value).abs()
        x, value, gradient = candidate, next_value, next_gradient
        change_done = change <= config.tolerance_change
        if config.termination == "change":
            done = change_done
        elif config.termination == "both":
            done = change_done & (torch.linalg.vector_norm(next_gradient, dim=1) <= config.tolerance_grad)
        elif config.termination == "gradient":
            done = torch.linalg.vector_norm(next_gradient, dim=1) <= config.tolerance_grad
        else:
            done = change_done | (torch.linalg.vector_norm(next_gradient, dim=1) <= config.tolerance_grad)
        newly = active & accepted & done
        converged |= newly
        iterations = torch.where((iterations == 0) & newly, torch.full_like(iterations, iteration), iterations)
        if not differentiable:
            x, value, gradient = x.detach(), value.detach(), gradient.detach()
        if config.record_debug:
            objective_history.append(value.detach().clone())
            gradient_history.append(torch.linalg.vector_norm(gradient.detach(), dim=1))
    iterations = torch.where(iterations == 0, torch.full_like(iterations, config.iterations), iterations)
    entry["last_solution"] = x.detach()
    final_norm = torch.linalg.vector_norm(gradient, dim=1)
    debug = None
    if config.record_debug:
        debug = {"objective": tuple(objective_history), "gradient_norm": tuple(gradient_history)}
    return _result(
        x, event_shape, batch_shape, value, converged, failed, iterations,
        final_gradient_norm=final_norm, debug=debug,
    )
