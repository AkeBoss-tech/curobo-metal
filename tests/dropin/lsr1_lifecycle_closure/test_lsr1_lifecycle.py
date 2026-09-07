from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt, jit_lsr1_compute_step_direction


class _Quadratic:
    action_horizon = 2
    action_dim = 1

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return (action - 1.5).square().sum(dim=(-1, -2))

    def update_batch_size(self, *, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)


def test_lsr1_batched_history_is_device_resident_and_no_worse() -> None:
    rollout = _Quadratic()
    optimizer = LSR1Opt(
        LBFGSOptCfg(num_iters=5, inner_iters=1, history=7, line_search_scale=[0.25, 0.5, 1.0]),
        [rollout, rollout],
    )
    seed = torch.tensor([[[0.0], [0.0]], [[3.0], [3.0]]])
    solved = optimizer.optimize(seed)
    assert (solved - 1.5).square().sum() < (seed - 1.5).square().sum()
    assert optimizer._s_history is not None and optimizer._y_history is not None
    # The action dimension is two, so the pinned history reduction retains one.
    assert optimizer._s_history.shape == (2, 1, 2)
    assert optimizer._s_history.device == seed.device
    assert optimizer._hessian_0 is not None and optimizer._hessian_0.shape == (2, 1, 1)
    assert rollout.batch_sizes == [6, 6]


def test_lsr1_reinitialize_mask_shift_and_reset_clear_rank_one_state() -> None:
    rollout = _Quadratic()
    optimizer = LSR1Opt(LBFGSOptCfg(num_iters=3, inner_iters=1), [rollout, rollout])
    seed = torch.zeros(2, 2, 1)
    optimizer.optimize(seed)
    assert optimizer._s_history is not None
    optimizer.reinitialize(torch.full_like(seed, 0.25), mask=torch.tensor([True, False]))
    assert torch.equal(optimizer._s_history[0], torch.zeros_like(optimizer._s_history[0]))
    optimizer.shift(1)
    assert optimizer._rho_history is not None
    assert torch.equal(optimizer._rho_history, torch.zeros_like(optimizer._rho_history))
    optimizer.reset()
    assert optimizer._s_history is None and optimizer._y_history is None and optimizer._hessian_0 is None


def test_lsr1_kernel_layout_and_validation_boundaries() -> None:
    grad = torch.tensor([[[2.0, -1.0]], [[1.0, 4.0]]])
    empty = torch.zeros(0, 2, 2, 1)
    direction = jit_lsr1_compute_step_direction(empty, empty, grad, 0, 1e-3, True, torch.ones(2, 1, 1))
    torch.testing.assert_close(direction, -grad)
    unstable_direction = jit_lsr1_compute_step_direction(
        empty, empty, grad, 0, 1e-3, False, torch.ones(2, 1, 1)
    )
    torch.testing.assert_close(unstable_direction, -grad)
    with pytest.raises(ValueError, match="history and gradient"):
        jit_lsr1_compute_step_direction(torch.zeros(2, 1, 3), torch.zeros(2, 1, 3), grad, 1, 1e-3, True, torch.ones(2, 1, 1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_lsr1_mps_uses_native_tensors_without_cpu_fallback() -> None:
    device = torch.device("mps")
    rollout = _Quadratic()
    optimizer = LSR1Opt(LBFGSOptCfg(num_iters=3, inner_iters=1), [rollout, rollout])
    solved = optimizer.optimize(torch.zeros(2, 2, 1, device=device))
    assert solved.device.type == "mps"
    assert torch.isfinite(solved).all()
