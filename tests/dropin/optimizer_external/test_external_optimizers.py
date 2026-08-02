"""Behavioral coverage for the portable external optimizer adapters."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.external._portable import UnsupportedExternalOptimizerFeature
from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.types.device_cfg import DeviceCfg


class _Collection:
    def __init__(self, cost, constraint=None):
        self._cost = cost
        self._constraint = constraint
        self.constraints = None if constraint is None else object()

    def get_sum_cost(self, **_):
        return self._cost

    def get_sum_constraint(self, **_):
        return self._constraint


class _Result:
    def __init__(self, collection):
        self.costs_and_constraints = collection


class _Rollout:
    action_horizon = 2
    action_dim = 2
    horizon = 2

    def __init__(self, device="cpu", constrained=False):
        self.action_bound_lows = torch.full((2,), -1.0, device=device)
        self.action_bound_highs = torch.full((2,), 1.0, device=device)
        self.constrained = constrained
        self.batch_sizes = []
        self.goal_dt = None

    def update_batch_size(self, batch_size):
        self.batch_sizes.append(batch_size)

    def update_goal_dt(self, value):
        self.goal_dt = value

    def evaluate_action(self, action):
        cost = (action - 0.4).square().sum(dim=-1, keepdim=True)
        violation = action.sum(dim=-1, keepdim=True) - 1.0 if self.constrained else None
        return _Result(_Collection(cost, violation))

    def compute_metrics_from_action(self, action):
        return {"action": action}


def test_torch_adapter_optimizes_rollout_batches_and_records_lifecycle():
    rollout = _Rollout()
    cfg = TorchOptCfg(
        num_iters=40,
        step_scale=0.08,
        torch_optim_name="SGD",
        store_debug=True,
        device_cfg=DeviceCfg(),
    )
    optimizer = TorchOpt(cfg, [rollout])
    seed = torch.zeros(2, 2, 2)
    result = optimizer.optimize(seed)
    assert result.shape == seed.shape
    assert (result - 0.4).square().sum() < (seed - 0.4).square().sum()
    assert optimizer.best_cost.shape == (2,)
    assert len(optimizer.get_recorded_trace()["debug"]) == cfg.num_iters + 1
    optimizer.update_goal_dt(0.02)
    assert rollout.goal_dt == 0.02
    assert optimizer.compute_metrics(result)["action"] is result


def test_torch_adapter_preserves_custom_optimizer_and_reinitializes_state():
    objective = lambda value: (value - 2.0).square().sum(dim=-1)
    optimizer = TorchOpt(
        TorchOptCfg(num_iters=12, step_scale=0.1, torch_optim_class=torch.optim.Adam), [objective]
    )
    output = optimizer.optimize(torch.zeros(1, 3))
    assert output.shape == (1, 3)
    assert torch.all(output > 0)
    optimizer.reinitialize(output, reset_num_iters=True)
    assert optimizer.best_cost.item() > 1e6


def test_scipy_adapter_has_a_device_resident_fallback_without_optional_scipy():
    """The locked Apple build intentionally has no SciPy dependency."""
    rollout = _Rollout()
    optimizer = ScipyOpt(ScipyOptCfg(num_iters=8, scipy_minimize_method="SLSQP"), [rollout])
    output = optimizer.optimize(torch.zeros(1, 2, 2))
    assert output.shape == (1, 2, 2)
    assert optimizer.last_result["backend"] == "torch-lbfgs"
    cost, gradient = optimizer._cost_constraint_and_gradient_fn_gpu(torch.zeros(4))
    assert cost.shape == (1,) and gradient.shape == (4,)


def test_scipy_adapter_uses_rollout_gradients_bounds_and_constraints_when_installed():
    pytest.importorskip("scipy")
    rollout = _Rollout(constrained=True)
    optimizer = ScipyOpt(
        ScipyOptCfg(num_iters=30, scipy_minimize_method="SLSQP", store_debug=True), [rollout]
    )
    output = optimizer.optimize(torch.zeros(1, 2, 2))
    assert output.shape == (1, 2, 2)
    assert bool((output <= 1.0 + 1e-6).all())
    assert bool((output.sum(dim=-1) <= 1.0 + 1e-4).all())
    assert getattr(optimizer.last_result, "success", True)
    cost, gradient = optimizer._cost_constraint_and_gradient_fn_gpu(torch.zeros(4))
    assert cost.shape == (1,) and gradient.shape == (4,)


def test_external_cuda_graph_request_is_explicitly_rejected():
    with pytest.raises(UnsupportedExternalOptimizerFeature, match="CUDA Graph"):
        TorchOpt(TorchOptCfg(), [lambda value: value.square().sum(-1)], use_cuda_graph=True)


def test_torch_adapter_runs_without_cpu_fallback_on_mps_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cfg = TorchOptCfg(
        num_iters=8,
        step_scale=0.1,
        torch_optim_name="SGD",
        device_cfg=DeviceCfg(device="mps", dtype=torch.float32),
    )
    result = TorchOpt(cfg, [lambda value: value.square().sum(dim=-1)]).optimize(
        torch.ones(1, 2, device="mps")
    )
    assert result.device.type == "mps"
