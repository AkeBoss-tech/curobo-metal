"""Behavioural regression coverage for the portable V2 gradient-descent facade."""

from __future__ import annotations

import torch
import pytest

from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.types.device_cfg import DeviceCfg


class _BoundedQuadratic:
    action_horizon = 2
    action_dim = 2
    horizon = 2
    action_bound_lows = torch.tensor([-0.25, -0.25])
    action_bound_highs = torch.tensor([0.25, 0.25])

    def __init__(self):
        self.metrics_calls = 0

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 0.8).square().sum(dim=(-1, -2))

    def compute_metrics_from_action(self, action: torch.Tensor):
        self.metrics_calls += 1
        return {"mean_action": action.mean()}


class _FlatConverged:
    action_horizon = 2
    action_dim = 1

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action.square().sum(dim=(-1, -2)) * 0.0


def test_gradient_descent_projects_bounds_and_accepts_flat_seed_inside_no_grad():
    rollout = _BoundedQuadratic()
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(num_iters=20, gradient_descent_step_scale=0.3), [rollout]
    )
    with torch.no_grad():
        output = optimizer.optimize(torch.full((1, 4), 2.0))
    assert output.shape == (1, 2, 2)
    assert bool((output <= 0.25).all())
    assert bool((output >= -0.25).all())
    assert output.requires_grad is False
    assert optimizer.solve_time >= 0.0
    assert optimizer.solver_names == ["gradient_descent"]
    assert optimizer.horizon == 2
    assert optimizer.get_all_rollout_instances() == [rollout]
    assert optimizer.compute_metrics(output)["mean_action"].item() == pytest.approx(0.25)
    assert rollout.metrics_calls == 1


def test_gradient_descent_nonfixed_convergence_and_lifecycle_reset_are_per_problem():
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(
            num_iters=20,
            fixed_iters=False,
            cost_convergence=1e-8,
            convergence_iteration=1,
            minimum_iters=2,
            converged_ratio=1.0,
            store_debug=True,
        ),
        [_FlatConverged()],
    )
    output = optimizer.optimize(torch.ones(1, 2, 1))
    torch.testing.assert_close(output, torch.ones_like(output))
    assert optimizer._iteration == 2
    assert optimizer._converged is not None and bool(optimizer._converged.all())
    assert len(optimizer.get_debug()["objective"]) == 2

    optimizer.update_niters(7)
    optimizer.reinitialize(output, reset_num_iters=True)
    assert optimizer.config.num_iters == 20
    assert optimizer._best_action is None
    assert optimizer.shift(1)
    optimizer.reset()
    assert optimizer._best_cost is None and optimizer.get_debug() is None


def test_gradient_descent_rejects_unsupported_rollout_particle_and_shape_contracts():
    with pytest.raises(ValueError, match="num_rollout_instances"):
        GradientDescentOpt(GradientDescentOptCfg(_num_rollout_instances=2), [_FlatConverged()])
    with pytest.raises(ValueError, match="num_particles=1"):
        GradientDescentOpt(GradientDescentOptCfg(num_particles=2), [_FlatConverged()])
    optimizer = GradientDescentOpt(GradientDescentOptCfg(), [_FlatConverged()])
    with pytest.raises(ValueError, match="seed_action"):
        optimizer.optimize(torch.zeros(2, 2, 1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_gradient_descent_runs_on_mps_without_cpu_fallback():
    rollout = _BoundedQuadratic()
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(
            num_iters=6,
            gradient_descent_step_scale=0.1,
            device_cfg=DeviceCfg(device="mps", dtype=torch.float32),
        ),
        [rollout],
    )
    output = optimizer.optimize(torch.zeros(1, 2, 2, device="mps"))
    assert output.device.type == "mps"
    assert optimizer._best_cost is not None and optimizer._best_cost.device.type == "mps"
