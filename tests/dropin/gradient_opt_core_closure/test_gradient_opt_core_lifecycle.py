from __future__ import annotations

import pytest
import torch

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.gradient_descent import GradientDescentOptCfg


class _Quadratic:
    action_horizon = 2
    action_dim = 1
    horizon = 2

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.initial_masks: list[torch.Tensor | None] = []

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 2.0).square().sum(dim=(-1, -2))

    def update_batch_size(self, *, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)


def _config(**kwargs) -> GradientDescentOptCfg:
    values = {"num_iters": 4, "inner_iters": 1, "step_scale": 1.0, "line_search_scale": [0.25, 0.5]}
    values.update(kwargs)
    config = GradientDescentOptCfg(**{key: value for key, value in values.items() if key != "line_search_scale"})
    config.line_search_scale = values["line_search_scale"]
    return config


def test_core_infers_initial_batch_and_noop_candidate_prevents_uphill_step() -> None:
    rollout = _Quadratic()
    core = GradientOptCore(_config(), [rollout], lambda state: -state.gradient)
    seed = torch.tensor([[[0.0], [0.0]], [[5.0], [5.0]]])
    solved = core.optimize(seed)
    assert solved.shape == seed.shape
    assert core.config.num_problems == 2
    assert rollout.batch_sizes == [2]
    assert (solved - 2.0).square().sum() < (seed - 2.0).square().sum()

    uphill = GradientOptCore(_config(num_iters=1), [_Quadratic()], lambda state: state.gradient)
    unchanged = uphill.optimize(torch.zeros(1, 2, 1))
    torch.testing.assert_close(unchanged, torch.zeros_like(unchanged))


def test_core_passes_mask_to_initial_hook_and_rejects_stale_warm_state() -> None:
    rollout = _Quadratic()
    masks: list[torch.Tensor | None] = []
    core = GradientOptCore(_config(num_problems=2), [rollout], lambda state: -state.gradient, on_initial_state=lambda _state, mask: masks.append(mask))
    seed = torch.zeros(2, 2, 1)
    core.reinitialize(seed, mask=torch.tensor([True, False]))
    assert masks and masks[-1] is not None
    torch.testing.assert_close(masks[-1], torch.tensor([True, False]))
    with pytest.raises(ValueError, match="must match"):
        core.optimize(torch.zeros(2, 2, 1, dtype=torch.float64))


def test_core_solver_update_is_atomic_and_reset_shape_discards_state() -> None:
    core = GradientOptCore(_config(), [_Quadratic()], lambda state: -state.gradient)
    old_iters = core.config.num_iters
    with pytest.raises(ValueError, match="multiple"):
        core.update_solver_params({"gradient_descent": {"num_iters": 3, "inner_iters": 2}})
    assert core.config.num_iters == old_iters
    core.reinitialize(torch.zeros(1, 2, 1))
    core.reset_shape()
    assert core._iteration_state is None and core._best_action is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_core_mps_state_remains_on_mps_without_fallback() -> None:
    core = GradientOptCore(_config(num_iters=2), [_Quadratic()], lambda state: -state.gradient)
    result = core.optimize(torch.zeros(2, 2, 1, device="mps"))
    assert result.device.type == "mps"
    assert core._best_action is not None and core._best_action.device.type == "mps"
