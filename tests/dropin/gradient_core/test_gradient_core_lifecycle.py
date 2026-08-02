"""Portable behavioural contract for :class:`GradientOptCore`."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.gradient_descent import GradientDescentOptCfg
from curobo._src.types.device_cfg import DeviceCfg


class _Quadratic:
    action_horizon = 2
    action_dim = 2
    horizon = 2

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 0.5).square().sum(dim=(-1, -2))


class _BoundedQuadratic(_Quadratic):
    action_bound_lows = torch.full((2,), -0.25)
    action_bound_highs = torch.full((2,), 0.25)


def _config(**kwargs):
    values = {
        "num_iters": 4,
        "step_scale": 0.2,
        "store_debug": True,
        "fixed_iters": True,
    }
    values.update(kwargs)
    return GradientDescentOptCfg(**values)


def test_core_optimizes_tracks_best_and_records_device_states():
    core = GradientOptCore(_config(), [_Quadratic()], lambda state: -state.gradient)
    seed = torch.full((1, 2, 2), 2.0)
    result = core.optimize(seed)

    assert result.shape == seed.shape
    assert (result - 0.5).square().sum() < (seed - 0.5).square().sum()
    trace = core.get_recorded_trace()
    assert len(trace["debug"]) == 5  # initial state plus four steps
    assert len(trace["debug_cost"]) == 5
    assert all(item.action.device == seed.device for item in trace["debug"])
    assert core._best_cost is not None and torch.isfinite(core._best_cost).all()


def test_core_projects_bounds_and_preserves_masked_reinitialization_state():
    calls: list[torch.Tensor | None] = []
    core = GradientOptCore(
        _config(num_problems=2, num_iters=1),
        [_BoundedQuadratic()],
        lambda state: -state.gradient,
        on_reinitialize=calls.append,
    )
    initial = torch.zeros(2, 2, 2)
    core.reinitialize(initial)
    assert core._iteration_state is not None
    first_state = core._iteration_state.action.clone()

    replacement = torch.full((2, 2, 2), 5.0)
    mask = torch.tensor([True, False])
    core.reinitialize(replacement, mask=mask)
    assert calls[0] is None
    torch.testing.assert_close(calls[1], mask)
    assert core._iteration_state is not None
    torch.testing.assert_close(core._iteration_state.action[0], torch.full((2, 2), 0.25))
    torch.testing.assert_close(core._iteration_state.action[1], first_state[1])


def test_core_validates_callback_and_solver_updates():
    core = GradientOptCore(_config(), [_Quadratic()], lambda _state: torch.zeros(1))
    with pytest.raises(ValueError, match="action-shaped"):
        core.optimize(torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="not found"):
        core.update_solver_params({"other": {}})
    assert core.update_solver_params({"gradient_descent": {"num_iters": 2, "inner_iters": 1}})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_core_runs_on_mps_without_cpu_fallback():
    config = _config(device_cfg=DeviceCfg(device="mps", dtype=torch.float32), num_iters=2)
    core = GradientOptCore(config, [_Quadratic()], lambda state: -state.gradient)
    result = core.optimize(torch.full((1, 2, 2), 2.0, device="mps"))
    assert result.device.type == "mps"
    assert core._best_action is not None and core._best_action.device.type == "mps"
