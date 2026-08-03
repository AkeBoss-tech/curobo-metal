"""Lifecycle coverage for portable V2 nonlinear conjugate gradient."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
    jit_cg_shift_buffers,
)
from curobo._src.optim.gradient.line_search_strategy import LineSearchType
from curobo._src.types.device_cfg import DeviceCfg


class _BoundedQuadratic:
    action_horizon = 3
    action_dim = 2
    horizon = 3
    action_bound_lows = torch.full((2,), -1.0)
    action_bound_highs = torch.full((2,), 1.0)

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 0.75).square().sum(dim=(-1, -2))


def _optimizer(*, line_search_type=LineSearchType.GREEDY, device_cfg=DeviceCfg(), **kwargs):
    return ConjugateGradientOpt(
        ConjugateGradientOptCfg(
            device_cfg=device_cfg,
            num_iters=8,
            num_problems=2,
            line_search_scale=[0.05, 0.2, 0.5, 1.0],
            line_search_type=line_search_type,
            store_debug=True,
            **kwargs,
        ),
        [_BoundedQuadratic()],
    )


@pytest.mark.parametrize(
    "kind",
    [
        LineSearchType.GREEDY,
        LineSearchType.ARMIJO,
        LineSearchType.WOLFE,
        LineSearchType.STRONG_WOLFE,
        LineSearchType.APPROX_WOLFE,
        LineSearchType.APPROX_STRONG_WOLFE,
    ],
)
def test_cg_line_search_modes_are_batched_bounded_and_reduce_cost(kind):
    optimizer = _optimizer(line_search_type=kind)
    seed = torch.zeros(2, 3, 2)
    solution = optimizer.optimize(seed)

    assert solution.shape == seed.shape
    assert bool((solution >= -1.0).all()) and bool((solution <= 1.0).all())
    assert (solution - 0.75).square().sum() < (seed - 0.75).square().sum()
    assert optimizer._prev_grad_q is not None and optimizer._prev_step is not None
    assert optimizer.solve_time >= 0.0
    assert len(optimizer.get_recorded_trace()["objective"]) >= 1


def test_cg_runs_under_no_grad_and_supports_flattened_seed_and_terminal_lock():
    optimizer = _optimizer(fix_terminal_action=True)
    seed = torch.zeros(2, 6)
    with torch.no_grad():
        output = optimizer.optimize(seed)

    assert output.shape == seed.shape
    # The final control remains at the seed value while earlier controls move.
    torch.testing.assert_close(output.reshape(2, 3, 2)[:, -1], torch.zeros(2, 2))
    assert bool((output[:, :-2] != 0.0).any())


def test_cg_state_reset_resize_and_horizon_shift_are_explicit():
    optimizer = _optimizer()
    seed = torch.zeros(2, 3, 2)
    optimizer.optimize(seed)
    before_gradient = optimizer._prev_grad_q.clone()
    before_step = optimizer._prev_step.clone()
    assert optimizer.shift(1)
    torch.testing.assert_close(optimizer._prev_grad_q[:, :-1], before_gradient[:, 1:])
    torch.testing.assert_close(optimizer._prev_step[:, :-1], before_step[:, 1:])
    torch.testing.assert_close(optimizer._prev_grad_q[:, -1], torch.zeros_like(before_gradient[:, -1]))
    optimizer.update_num_problems(3)
    assert optimizer._prev_grad_q is optimizer._prev_step is None
    optimizer.reinitialize(torch.zeros(3, 3, 2))
    assert optimizer._prev_grad_q is optimizer._prev_step is None
    optimizer.reset()
    assert optimizer._prev_grad_q is optimizer._prev_step is None


def test_cg_zero_shift_is_identity_and_configuration_validation_is_strict():
    previous = torch.arange(12.0).reshape(2, 1, 6)
    shifted_gradient, shifted_step = jit_cg_shift_buffers(previous, -previous, 0, 2)
    torch.testing.assert_close(shifted_gradient, previous)
    torch.testing.assert_close(shifted_step, -previous)
    with pytest.raises(ValueError, match="finite nonnegative"):
        ConjugateGradientOptCfg(line_search_scale=[float("nan")])
    with pytest.raises(ValueError, match="line_search_wolfe_c_2"):
        ConjugateGradientOptCfg(line_search_wolfe_c_2=1.1)
    with pytest.raises(TypeError, match="fix_terminal_action"):
        ConjugateGradientOptCfg(fix_terminal_action="yes")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_cg_line_search_executes_on_mps_without_fallback():
    device_cfg = DeviceCfg(device="mps", dtype=torch.float32)
    optimizer = _optimizer(device_cfg=device_cfg, line_search_type=LineSearchType.STRONG_WOLFE)
    output = optimizer.optimize(torch.zeros(2, 3, 2, device="mps"))
    assert output.device.type == "mps"
    assert optimizer._prev_grad_q.device.type == optimizer._prev_step.device.type == "mps"
