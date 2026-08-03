"""Batched portable L-BFGS history and lifecycle regression coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg


class _Quadratic:
    action_horizon = 2
    action_dim = 1
    horizon = 2

    def __init__(self, target: float = 2.0) -> None:
        self.target = target
        self.batch_sizes: list[int] = []

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.target).square().sum(dim=(-1, -2))

    def update_batch_size(self, *, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)


def test_two_loop_history_is_per_problem_and_improves_batched_quadratics() -> None:
    rollout = _Quadratic()
    optimizer = LBFGSOpt(
        LBFGSOptCfg(num_iters=4, inner_iters=2, line_search_scale=[0.25, 0.5, 1.0], history=3),
        [rollout, rollout],
    )
    seed = torch.tensor([[[0.0], [0.0]], [[4.0], [4.0]]])
    result = optimizer.optimize(seed)
    assert (result - 2.0).square().sum() < (seed - 2.0).square().sum()
    assert optimizer._s_history is not None and optimizer._s_history.shape[:2] == (2, 3)
    assert optimizer._y_history is not None and optimizer._rho_history is not None
    assert bool((optimizer._rho_history >= 0).all())
    assert optimizer._best_action is not None and optimizer._best_action.device == seed.device
    # Direct multi-batch use resized the portable rollout cache deterministically.
    # Both configured rollout instances receive the same expanded candidate
    # batch size.  This fixture intentionally uses one object twice.
    assert rollout.batch_sizes == [6, 6]


def test_masked_warm_start_terminal_lock_and_lifecycle_reset() -> None:
    rollout = _Quadratic()
    optimizer = LBFGSOpt(
        LBFGSOptCfg(num_problems=2, num_iters=2, inner_iters=1, line_search_scale=[0.5, 1.0]),
        [rollout, rollout],
    )
    optimizer.reinitialize(torch.tensor([[[1.0], [1.0]], [[5.0], [5.0]]]))
    optimizer.reinitialize(
        torch.tensor([[[0.0], [0.0]], [[9.0], [9.0]]]), mask=torch.tensor([True, False])
    )
    warmed = optimizer.optimize(torch.zeros((2, 2, 1)))
    assert warmed.shape == (2, 2, 1)
    assert optimizer._pending_action is None

    locked = LBFGSOpt(
        LBFGSOptCfg(num_iters=4, inner_iters=2, fix_terminal_action=True, line_search_scale=[0.5, 1.0]),
        [_Quadratic(), _Quadratic()],
    )
    output = locked.optimize(torch.tensor([[[0.0], [1.0]]]))
    assert output[0, 0, 0] > 0.0
    assert output[0, 1, 0] == pytest.approx(1.0)
    locked.reset()
    assert locked._s_history is None and locked._best_action is None


def test_config_and_runtime_updates_are_validated_without_partial_mutation() -> None:
    with pytest.raises(ValueError, match="history"):
        LBFGSOptCfg(history=0)
    with pytest.raises(ValueError, match="Wolfe"):
        LBFGSOptCfg(line_search_wolfe_c_1=0.9, line_search_wolfe_c_2=0.1)
    optimizer = LBFGSOpt(LBFGSOptCfg(num_iters=4, inner_iters=2), [_Quadratic(), _Quadratic()])
    with pytest.raises(ValueError, match="unknown optimizer"):
        optimizer.update_solver_params({"lbfgs": {"missing": 1}})
    with pytest.raises(ValueError, match="history"):
        optimizer.update_solver_params({"lbfgs": {"history": 0}})
    assert optimizer.config.history == 7
    optimizer.update_num_problems(3)
    assert optimizer.config.num_problems == 3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_lbfgs_history_and_candidate_search_stay_on_mps_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    rollout = _Quadratic()
    optimizer = LBFGSOpt(
        LBFGSOptCfg(num_iters=3, inner_iters=1, line_search_scale=[0.25, 0.5, 1.0]),
        [rollout, rollout],
    )
    output = optimizer.optimize(torch.zeros((2, 2, 1), device="mps"))
    assert output.device.type == "mps"
    assert optimizer._s_history is not None and optimizer._s_history.device.type == "mps"
    assert optimizer._best_cost is not None and optimizer._best_cost.device.type == "mps"
