"""Executable portable behavior for pinned V2 quasi-Newton helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
    jit_cg_compute_step_direction,
    jit_cg_shift_buffers,
)
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.gradient.line_search_strategy import (
    ApproxStrongWolfeLineSearchStrategy,
    ArmijoLineSearchStrategy,
    GreedyLineSearchStrategy,
    LineSearchStrategyFactory,
    LineSearchType,
    StrongWolfeLineSearchStrategy,
    WolfeLineSearchStrategy,
)
from curobo._src.optim.gradient.lsr1 import LSR1Opt, jit_lsr1_compute_step_direction
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class _QuadraticRollout:
    action_horizon = 1
    action_dim = 2

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 2.0).square().sum(dim=(-1, -2))


def _context(target: float = 0.8):
    def evaluate(points: torch.Tensor):
        delta = points - target
        return delta.square().sum(dim=(-1, -2)), 2.0 * delta

    return SimpleNamespace(
        line_search_scale=torch.tensor([0.1, 0.5, 1.0]),
        line_search_c_1=1e-4,
        line_search_c_2=0.9,
        step_scale=1.0,
        fix_terminal_action=False,
        action_horizon=1,
        action_horizon_step_max=torch.ones(1, 1),
        compute_costs_and_gradients=evaluate,
    )


def _state(target: float = 0.8):
    action = torch.zeros(2, 1, 1)
    return OptimizationIterationState(
        action=action,
        exploration_action=action,
        exploration_cost=torch.full((2,), target * target),
        exploration_gradient=torch.full_like(action, -2.0 * target),
        step_direction=torch.ones_like(action),
    )


@pytest.mark.parametrize(
    "strategy",
    [
        GreedyLineSearchStrategy(), ArmijoLineSearchStrategy(), WolfeLineSearchStrategy(),
        StrongWolfeLineSearchStrategy(), ApproxStrongWolfeLineSearchStrategy(),
    ],
)
def test_line_searches_keep_batched_device_tensor_results(strategy):
    result = strategy.search(_state(), _context())
    assert result.selected_state.action.shape == (2, 1, 1)
    assert result.selected_state.cost.shape == (2,)
    assert result.selected_state.idxs.shape == (2,)
    assert torch.isfinite(result.selected_state.action).all()
    # The largest candidate is the closest one to 0.8 and must be selected.
    torch.testing.assert_close(result.selected_state.action, torch.ones(2, 1, 1))


def test_line_search_factory_and_terminal_step_boundary():
    assert isinstance(LineSearchStrategyFactory.get_strategy("APPROX_WOLFE"), WolfeLineSearchStrategy)
    assert LineSearchType("STRONG_WOLFE") is LineSearchType.STRONG_WOLFE
    delta = torch.tensor([[[3.0], [2.0]]])
    scaled = GreedyLineSearchStrategy.scale_action(delta, torch.ones(2, 1), 0.5, True, 2)
    torch.testing.assert_close(scaled, torch.tensor([[[1.0], [0.0]]]))
    with pytest.raises(ValueError, match="already registered"):
        LineSearchStrategyFactory.register_strategy(LineSearchType.GREEDY, GreedyLineSearchStrategy)


def test_cg_formulas_shift_and_optimizer_reduce_quadratic_cost():
    grad = torch.tensor([[[2.0, 1.0]]])
    previous = torch.tensor([[[1.0, 1.0]]])
    step = -previous.clone()
    direction, prev_grad, prev_step = jit_cg_compute_step_direction(grad, previous, step, 10.0, "PR")
    assert direction.shape == grad.shape
    torch.testing.assert_close(prev_grad, grad)
    torch.testing.assert_close(prev_step, direction)
    shifted_grad, shifted_step = jit_cg_shift_buffers(previous, step, 1, 1)
    torch.testing.assert_close(shifted_grad, torch.tensor([[[1.0, 0.0]]]))
    torch.testing.assert_close(shifted_step, torch.tensor([[[-1.0, 0.0]]]))

    optimizer = ConjugateGradientOpt(
        ConjugateGradientOptCfg(num_iters=12, inner_iters=3, line_search_scale=[0.1, 0.5, 1.0]),
        [_QuadraticRollout()],
    )
    seed = torch.zeros(3, 1, 2)
    solved = optimizer.optimize(seed)
    assert (solved - 2.0).square().sum() < (seed - 2.0).square().sum()
    optimizer.shift(1)
    assert optimizer._prev_grad_q is not None


def test_lsr1_direction_and_optimizer_reduce_quadratic_cost():
    grad = torch.tensor([[[2.0, -1.0]]])
    direction = jit_lsr1_compute_step_direction(
        torch.zeros(1, 0, 2), torch.zeros(1, 0, 2), grad, 7, 1e-2, True, torch.ones(1, 1, 1)
    )
    torch.testing.assert_close(direction, -grad)
    optimizer = LSR1Opt(
        LBFGSOptCfg(num_iters=15, inner_iters=5, line_search_scale=[0.1, 0.5, 1.0]),
        [_QuadraticRollout(), _QuadraticRollout()],
    )
    seed = torch.zeros(2, 1, 2)
    solved = optimizer.optimize(seed)
    assert (solved - 2.0).square().sum() < (seed - 2.0).square().sum()
    assert optimizer._s_history is not None


def test_lbfgs_config_normalizes_line_search_and_rejects_cuda_kernel_flags():
    config = LBFGSOptCfg(line_search_type="STRONG_WOLFE", use_cuda_kernel_line_search=True)
    assert config.line_search_type is LineSearchType.STRONG_WOLFE
    assert not config.use_cuda_kernel_line_search
    with pytest.raises(ValueError, match="multiple"):
        config.update_niters(26)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_quasi_newton_cpu_independent_mps_smoke_without_fallback():
    device = torch.device("mps")
    optimizer = ConjugateGradientOpt(
        ConjugateGradientOptCfg(num_iters=8, inner_iters=2, line_search_scale=[0.25, 0.5, 1.0]),
        [_QuadraticRollout()],
    )
    seed = torch.zeros(2, 1, 2, device=device)
    solution = optimizer.optimize(seed)
    assert solution.device.type == "mps"
    assert torch.isfinite(solution).all()
