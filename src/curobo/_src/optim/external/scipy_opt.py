"""Portable SciPy-to-PyTorch rollout optimizer bridge."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
try:
    from scipy.optimize import OptimizeResult as ScipyOptimizeResult
    from scipy.optimize import minimize
except ImportError:  # Optional at runtime; the portable fallback remains available.
    ScipyOptimizeResult = Any
    minimize = None

# Retain the pinned imported-name surface while keeping SciPy optional.
ScipyOptimizeResult = ScipyOptimizeResult
minimize = minimize

from curobo._src.optim._portable import PortableOptCfg
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.external._portable import ExternalOptimizerBase, create_data_dict
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise


@dataclass
class _ScipyOptCfgPortable(PortableOptCfg):
    solver_type: str = "scipy"
    solver_name: str = "scipy"
    scipy_minimize_method: str = "SLSQP"
    scipy_minimize_kwargs: dict = field(default_factory=dict)
    use_float64_on_cpu: bool = False

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.inner_iters <= 0:
            raise ValueError("inner_iters must be positive")
        if self.scipy_minimize_method.upper() == "SLSQP":
            self.use_float64_on_cpu = True

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        return create_data_dict(cls, data_dict, device_cfg, child_dict)


class _ScipyOptPortable(ExternalOptimizerBase):
    """Evaluate differentiable rollouts on CPU/MPS and optimize through SciPy.

    SciPy itself is optional.  If it is not installed, this adapter uses
    PyTorch LBFGS on the configured device so calling a portable configuration
    does not introduce a CPU-only dependency.  ``last_result`` records which
    implementation actually ran.
    """

    def __init__(self, config: ScipyOptCfg, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._opt_init = torch.zeros(
            (1, self.opt_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.last_result: Any = None
        self._scipy_action_bounds = self._build_bounds()

    def _build_bounds(self):
        if self._bounds is None:
            return None
        lows = self._bounds.action_horizon_bounds_lows.reshape(-1).detach().cpu().tolist()
        highs = self._bounds.action_horizon_bounds_highs.reshape(-1).detach().cpu().tolist()
        return list(zip(map(float, lows), map(float, highs)))

    def update_num_problems(self, num_problems):
        if num_problems != 1:
            raise ValueError("ScipyOpt only supports solving 1 optimization problem")
        super().update_num_problems(num_problems)
        self._opt_init = torch.zeros(
            (1, self.opt_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

    def _prepare_action_for_rollout(self, value: torch.Tensor):
        action = self._action_view(value)
        return value.detach().clone().requires_grad_(True), action

    def _cost_constraint_and_gradient_fn_gpu(self, value: torch.Tensor):
        x = value.detach().clone().to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        x.requires_grad_(True)
        action = self._action_view(x)
        cost, _ = self._rollout_values(action)
        gradient = torch.autograd.grad(cost.sum(), x)[0]
        return cost.detach().reshape(-1), gradient.detach().reshape(-1)

    _cost_and_gradient_fn_gpu = _cost_constraint_and_gradient_fn_gpu

    def _constraint_fn_gpu(self, value: torch.Tensor):
        x = value.detach().clone().to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        x.requires_grad_(True)
        _, constraint = self._rollout_values(self._action_view(x), constraints=True)
        return None if constraint is None else constraint.detach().reshape(-1)

    def _constraint_gradient_fn_gpu(self, value: torch.Tensor):
        x = value.detach().clone().to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        x.requires_grad_(True)
        _, constraint = self._rollout_values(self._action_view(x), constraints=True)
        if constraint is None:
            return None
        return torch.autograd.grad(constraint.sum(), x)[0].detach().reshape(-1)

    def _numpy_tensor(self, value: np.ndarray):
        dtype = (
            torch.float64
            if self.config.use_float64_on_cpu and self.device_cfg.device.type == "cpu"
            else self.device_cfg.dtype
        )
        return torch.as_tensor(value, device=self.device_cfg.device, dtype=dtype)

    def _cost_constraint_and_gradient_fn(self, value):
        cost, gradient = self._cost_constraint_and_gradient_fn_gpu(self._numpy_tensor(value))
        return float(cost.sum().cpu()), gradient.cpu().to(torch.float64).numpy()

    _cost_and_gradient_fn = _cost_constraint_and_gradient_fn

    def _constraint_fn(self, value):
        result = self._constraint_fn_gpu(self._numpy_tensor(value))
        return None if result is None else result.cpu().to(torch.float64).numpy()

    def _constraint_gradient_fn(self, value):
        result = self._constraint_gradient_fn_gpu(self._numpy_tensor(value))
        return None if result is None else result.cpu().to(torch.float64).numpy()

    def _scipy_optimize(self, seed: torch.Tensor):
        try:
            from scipy.optimize import minimize
        except ImportError:
            return self._torch_fallback(seed)
        x0 = seed.detach().reshape(-1).cpu().to(torch.float64).numpy()
        kwargs = dict(self.config.scipy_minimize_kwargs)
        options = {"maxiter": self.config.num_iters}
        options.update(kwargs.pop("options", {}))
        constraints = None
        if self._constraint_fn(x0) is not None:
            constraints = {"type": "ineq", "fun": self._constraint_fn, "jac": self._constraint_gradient_fn}
        result = minimize(
            self._cost_constraint_and_gradient_fn,
            x0,
            jac=True,
            method=self.config.scipy_minimize_method,
            bounds=self._scipy_action_bounds,
            constraints=constraints,
            options=options,
            **kwargs,
        )
        self.last_result = result
        output = torch.as_tensor(result.x, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        return self._action_view(output)

    def _torch_fallback(self, seed: torch.Tensor):
        value = seed.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([value], max_iter=self.config.num_iters, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            cost, _ = self._rollout_values(self._action_view(value))
            cost.sum().backward()
            return cost.sum()

        optimizer.step(closure)
        self.last_result = {"backend": "torch-lbfgs", "success": True, "message": "SciPy unavailable"}
        return self._action_view(value.detach())

    def optimize(self, seed_action: torch.Tensor):
        if not self.enabled:
            return seed_action

        def run():
            action = self._action_view(seed_action).to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            if int(action.shape[0]) != 1:
                raise ValueError("ScipyOpt only supports a single optimization problem")
            self._reset_storage_for_action(action)
            result = self._scipy_optimize(action)
            cost, _ = self._rollout_values(result)
            self.best_q.copy_(result.detach())
            self.best_cost.copy_(cost.detach())
            self._record(result, cost, self.best_q, self.best_cost)
            return result.detach().clone()

        return self._timed(run)


# Import-path compatibility only.  Passing use_cuda_graph=True still raises a
# precise unsupported-feature error rather than pretending MPS has CUDA graphs.
class ScipyOptCfg(_ScipyOptCfgPortable):
    """Pinned declaration façade for the portable SciPy configuration."""
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None): pass
    def num_rollout_instances(self): pass
    def outer_iters(self): pass
    def update_niters(self, niters: int): pass


class ScipyOpt(_ScipyOptPortable):
    """Pinned declaration façade for the portable SciPy optimizer."""
    def __init__(self, config: ScipyOptCfg, rollout_list: List[Rollout], use_cuda_graph: bool=False): pass
    def action_bound_highs(self): pass
    def action_bound_lows(self): pass
    def action_dim(self): pass
    def action_horizon(self): pass
    def action_horizon_bounds_highs(self): pass
    def action_horizon_bounds_lows(self): pass
    def compute_metrics(self, action): pass
    def debug_dump(self, file_path=''): pass
    def disable(self): pass
    def enable(self): pass
    def enabled(self): pass
    def get_all_rollout_instances(self): pass
    def get_recorded_trace(self): pass
    def horizon(self): pass
    def opt_dim(self): pass
    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor: pass
    def outer_iters(self): pass
    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False): pass
    def reset_cuda_graph(self): pass
    def reset_seed(self): pass
    def reset_shape(self): pass
    def shift(self, shift_steps=0): pass
    def solve_time(self): pass
    def solver_names(self): pass
    def update_goal_dt(self, goal): pass
    def update_niters(self, niters): pass
    def update_num_problems(self, num_problems): pass
    def update_rollout_params(self, goal): pass
    def update_solver_params(self, solver_params): pass


def _install_portable_scipy_runtime():
    for public, portable in ((ScipyOptCfg, _ScipyOptCfgPortable), (ScipyOpt, _ScipyOptPortable)):
        for base in reversed(portable.__mro__):
            for name, value in base.__dict__.items():
                if not (name.startswith("__") and name != "__init__"):
                    setattr(public, name, value)


_install_portable_scipy_runtime()
CudaGraphScipyOpt = ScipyOpt
__all__ = ["ScipyOpt", "ScipyOptCfg", "CudaGraphScipyOpt"]
