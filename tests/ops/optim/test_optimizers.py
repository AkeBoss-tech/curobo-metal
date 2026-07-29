from __future__ import annotations

import os

import pytest
import torch

from curobo_metal.optim import (
    ExecutionCache, LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize,
)


def sphere(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lbfgs_batched_quadratic_and_per_seed_results(dtype: torch.dtype) -> None:
    initial = torch.tensor([[3.0, -2.0], [0.0, 0.0], [-4.0, 1.0]], dtype=dtype)
    result = lbfgs_optimize(
        sphere, initial,
        config=LBFGSConfig(iterations=30, tolerance_grad=1e-6, tolerance_change=1e-12),
    )
    assert result.solution.shape == initial.shape
    assert torch.allclose(result.solution, torch.zeros_like(initial), atol=2e-5, rtol=0)
    assert result.converged.tolist() == [True, True, True]
    assert not bool(result.failed.any())
    assert result.iterations.shape == (3,)


def test_lbfgs_projection_gradient_and_nonfinite_failure() -> None:
    target = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    result = lbfgs_optimize(
        lambda x: (x - target).square().sum(-1),
        torch.zeros_like(target),
        projection=lambda x: x.clamp(-1, 1),
        config=LBFGSConfig(iterations=8),
        differentiable=True,
    )
    assert torch.allclose(result.solution, torch.ones_like(result.solution))
    assert torch.autograd.grad(result.solution.sum(), target, allow_unused=True)[0] is not None
    failed = lbfgs_optimize(
        lambda x: torch.full(x.shape[:-1], torch.nan, dtype=x.dtype),
        torch.zeros((2, 1), dtype=torch.float64),
    )
    assert failed.failed.tolist() == [True, True]


def test_particle_is_deterministic_batched_and_improves() -> None:
    initial = torch.tensor([[3.0, -2.0], [-4.0, 1.0]])
    config = ParticleConfig(
        iterations=20, particles=48, elite_count=8, initial_std=2.0, seed=17,
    )
    first = particle_optimize(sphere, initial, config=config)
    second = particle_optimize(sphere, initial, config=config)
    assert torch.equal(first.solution, second.solution)
    assert bool((first.objective < sphere(initial)).all())
    assert first.solution.shape == initial.shape


def test_mppi_covariance_debug_and_lbfgs_termination_state() -> None:
    initial = torch.tensor([[2.0, -1.0]])
    particle = particle_optimize(
        sphere, initial,
        config=ParticleConfig(
            iterations=4, particles=16, elite_count=4, strategy="mppi",
            covariance=torch.tensor([1.0, 0.25]), record_debug=True,
        ),
    )
    assert particle.debug is not None and len(particle.debug["objective"]) == 4
    gradient = lbfgs_optimize(
        sphere, initial,
        config=LBFGSConfig(line_search="none", termination="gradient", record_debug=True),
    )
    assert gradient.final_gradient_norm is not None
    assert gradient.debug is not None


@pytest.mark.parametrize("solver", ["lbfgs", "particle"])
def test_empty_batches(solver: str) -> None:
    initial = torch.empty((0, 3))
    result = (
        lbfgs_optimize(sphere, initial)
        if solver == "lbfgs"
        else particle_optimize(sphere, initial)
    )
    assert result.solution.shape == (0, 3)
    assert result.objective.shape == (0,)
    assert result.status == ()


def test_shape_keyed_cache_warm_start_reset_and_update() -> None:
    cache = ExecutionCache(capacity=2)
    config = LBFGSConfig(iterations=4)
    first = lbfgs_optimize(sphere, torch.ones((2, 3)), config=config, cache=cache)
    assert cache.misses == 1 and cache.size == 1
    warm = lbfgs_optimize(
        sphere, torch.full((2, 3), 9.0), config=config, cache=cache, warm_start=True,
    )
    assert cache.hits == 1
    assert bool((warm.objective <= first.objective + 1e-8).all())
    generation = cache.generation
    cache.reset()
    assert cache.size == 0 and cache.generation == generation + 1
    cache.update(capacity=0)
    assert cache.capacity == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_with_fallback_disabled() -> None:
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") in ("0", "")
    initial = torch.tensor([[2.0, -1.0]], device="mps")
    result = lbfgs_optimize(sphere, initial, config=LBFGSConfig(iterations=12))
    assert result.solution.device.type == "mps"
    assert float(result.objective.item()) < 1e-6
