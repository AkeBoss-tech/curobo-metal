"""Portable V2 line-search lifecycle and autograd coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_strategy import (
    ApproxStrongWolfeLineSearchStrategy,
    ApproxWolfeLineSearchStrategy,
    ArmijoLineSearchStrategy,
    GreedyLineSearchStrategy,
    LineSearchStrategy,
    LineSearchStrategyFactory,
    LineSearchType,
    StrongWolfeLineSearchStrategy,
    WolfeLineSearchStrategy,
)
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg


def _context(*, device_cfg=DeviceCfg(), scales=(0.0, 0.25, 0.5, 1.0)):
    target = torch.tensor(0.8, device=device_cfg.device, dtype=device_cfg.dtype)

    def evaluate(points: torch.Tensor):
        cost = (points - target).square().sum(dim=(-1, -2))
        gradient = torch.autograd.grad(cost.sum(), points)[0]
        return cost, gradient

    return LineSearchContext(
        device_cfg=device_cfg,
        line_search_scale=list(scales),
        line_search_c_1=1.0e-4,
        line_search_c_2=0.9,
        num_problems=2,
        opt_dim=2,
        action_horizon=1,
        action_dim=2,
        step_scale=1.0,
        fix_terminal_action=False,
        action_horizon_step_max=torch.ones(1, 2, device=device_cfg.device, dtype=device_cfg.dtype),
        use_cuda_kernel_line_search=False,
        compute_costs_and_gradients=evaluate,
        convergence_iteration=0,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )


def _state(*, device="cpu"):
    action = torch.zeros(2, 1, 2, device=device)
    return OptimizationIterationState(
        action=action,
        exploration_action=action,
        exploration_cost=torch.full((2,), 1.28, device=device),
        exploration_gradient=torch.full_like(action, -1.6),
        step_direction=torch.ones_like(action),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        GreedyLineSearchStrategy(),
        ArmijoLineSearchStrategy(),
        WolfeLineSearchStrategy(),
        StrongWolfeLineSearchStrategy(),
        ApproxWolfeLineSearchStrategy(),
        ApproxStrongWolfeLineSearchStrategy(),
    ],
)
def test_line_searches_create_autograd_candidates_inside_no_grad_and_keep_batches(strategy):
    context = _context()
    state = _state()
    with torch.no_grad():
        result = strategy.search(state, context)

    assert result.selected_state.action.shape == (2, 1, 2)
    assert result.exploration_state.action.shape == (2, 1, 2)
    assert result.selected_state.idxs.shape == (2,)
    assert not result.selected_state.action.requires_grad
    assert bool((result.selected_state.idxs >= 0).all())
    assert bool((result.selected_state.idxs < context.n_linesearch).all())


def test_line_search_resizing_and_scale_boundaries_are_explicit():
    context = _context()
    strategy = GreedyLineSearchStrategy()
    assert strategy.update_num_problems(3, context) is strategy
    assert strategy.num_problems == context.num_problems == 3
    with pytest.raises(ValueError, match="num_problems"):
        strategy.update_num_problems(True, context)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LineSearchStrategy.jit_get_x_set(torch.ones(1, 1, 1), torch.zeros(1, 1, 1), [float("nan")])
    with pytest.raises(ValueError, match="matching"):
        LineSearchStrategy.jit_get_x_set(torch.ones(1, 1, 1), torch.zeros(1, 2, 1), [1.0])
    with pytest.raises(ValueError, match="action_horizon"):
        LineSearchStrategy.scale_action(torch.ones(1, 2, 1), None, 1.0, False, 1)
    with pytest.raises(TypeError, match="fix_terminal_action"):
        LineSearchStrategy.scale_action(torch.ones(1, 1, 1), None, 1.0, 1, 1)


def test_line_search_rejects_malformed_context_tensors_and_preserves_first_ties():
    context = _context(scales=(0.0, 1.0))
    state = _state()
    tie = GreedyLineSearchStrategy()

    def flat_cost(points: torch.Tensor):
        return torch.zeros(points.shape[:2], device=points.device), torch.zeros_like(points)

    context.compute_costs_and_gradients = flat_cost
    result = tie.search(state, context)
    assert result.selected_state.idxs.tolist() == [0, 0]

    context.compute_costs_and_gradients = lambda points: (
        torch.zeros(points.shape[:2], device=points.device),
        torch.zeros(points.shape[0], points.shape[1], 1, 1, device=points.device),
    )
    with pytest.raises(ValueError, match="candidate-point shape"):
        tie.search(state, context)


def test_line_search_factory_parses_all_pinned_names_and_rejects_unknowns():
    for kind in LineSearchType:
        assert isinstance(LineSearchStrategyFactory.get_strategy(kind.name), LineSearchStrategy)
    with pytest.raises(ValueError, match="Unknown line search type"):
        LineSearchStrategyFactory.get_strategy("cubic")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_line_search_autograd_executes_on_mps_without_fallback():
    device_cfg = DeviceCfg(device="mps", dtype=torch.float32)
    result = ApproxStrongWolfeLineSearchStrategy().search(_state(device="mps"), _context(device_cfg=device_cfg))
    assert result.selected_state.action.device.type == "mps"
    assert result.selected_state.gradient.device.type == "mps"
