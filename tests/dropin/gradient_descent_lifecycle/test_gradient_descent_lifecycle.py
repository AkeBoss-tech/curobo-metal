"""Lifecycle and batch-state coverage for portable gradient descent."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg


class _Quadratic:
    action_horizon = 1
    action_dim = 1
    horizon = 1

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action.square().sum(dim=(-1, -2))

    def update_batch_size(self, *, batch_size: int) -> None:
        self.batch_sizes.append(batch_size)


class _ConstantObjective:
    action_horizon = 1
    action_dim = 1

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        # Deliberately independent of action: the portable optimizer should
        # treat its gradient as zero rather than ask autograd for an invalid VJP.
        return torch.full((action.shape[0],), 2.0, device=action.device, dtype=action.dtype)


def test_masked_reinitialize_is_consumed_by_next_batched_solve() -> None:
    rollout = _Quadratic()
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(
            num_problems=2,
            num_iters=1,
            gradient_descent_step_scale=0.0,
        ),
        [rollout],
    )
    optimizer.reinitialize(torch.tensor([[[5.0]], [[5.0]]]))
    optimizer.reinitialize(
        torch.tensor([[[1.0]], [[9.0]]]), mask=torch.tensor([True, False])
    )
    # The new action applies only to selected problem zero; problem one keeps
    # its earlier warm start despite an unrelated incoming solve seed.
    output = optimizer.optimize(torch.zeros((2, 1, 1)))
    torch.testing.assert_close(output[..., 0], torch.tensor([[1.0], [5.0]]))
    assert optimizer._reinitialized_action is None
    assert optimizer._best_action is not None


def test_constant_objective_and_resize_have_portable_lifecycle() -> None:
    constant = GradientDescentOpt(
        GradientDescentOptCfg(num_iters=3, store_debug=True), [_ConstantObjective()]
    )
    seed = torch.tensor([[[3.0]]])
    torch.testing.assert_close(constant.optimize(seed), seed)
    assert constant._iteration == 3
    assert len(constant.get_debug()["objective"]) == 3

    rollout = _Quadratic()
    optimizer = GradientDescentOpt(GradientDescentOptCfg(num_iters=1), [rollout])
    optimizer.update_num_problems(3)
    assert optimizer.config.num_problems == 3
    assert rollout.batch_sizes == [3]
    result = optimizer.optimize(torch.ones((3, 1, 1)))
    assert result.shape == (3, 1, 1)
    with pytest.raises(ValueError, match="num_problems"):
        optimizer.update_num_problems(0)


def test_config_and_runtime_parameter_updates_are_atomic_and_validated() -> None:
    with pytest.raises(ValueError, match="step scales"):
        GradientDescentOptCfg(gradient_descent_step_scale=-0.1)
    with pytest.raises(ValueError, match="num_problems"):
        GradientDescentOptCfg(num_problems=0)
    with pytest.raises(ValueError, match="convergence_iteration"):
        GradientDescentOptCfg(convergence_iteration=-1)

    optimizer = GradientDescentOpt(GradientDescentOptCfg(num_iters=4), [_Quadratic()])
    with pytest.raises(ValueError, match="unknown optimizer"):
        optimizer.update_solver_params({"gradient_descent": {"unknown": 1}})
    with pytest.raises(ValueError, match="num_iters"):
        optimizer.update_solver_params({"gradient_descent": {"num_iters": 0}})
    assert optimizer.config.num_iters == 4
    assert optimizer.update_solver_params({"gradient_descent": {"num_iters": 2}})
    assert optimizer.config.num_iters == 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_reinitialize_and_resize_stay_on_mps_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(num_problems=2, num_iters=1, gradient_descent_step_scale=0.0),
        [_Quadratic()],
    )
    optimizer.reinitialize(torch.ones((2, 1, 1), device="mps"))
    optimizer.reinitialize(torch.zeros((2, 1, 1), device="mps"), mask=torch.tensor([True, False], device="mps"))
    output = optimizer.optimize(torch.full((2, 1, 1), 3.0, device="mps"))
    assert output.device.type == "mps"
    torch.testing.assert_close(output[..., 0], torch.tensor([[0.0], [1.0]], device="mps"))
