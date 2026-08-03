"""Pinned L-BFGS surface backed by portable batched L-BFGS."""

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional

import torch

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer, _objective
from curobo._src.types.device_cfg import DeviceCfg
from .line_search_strategy import LineSearchType


@dataclass
class LBFGSOptCfg(PortableOptCfg):
    solver_type: str = "lbfgs"
    solver_name: str = "lbfgs"
    inner_iters: int = 25
    _num_rollout_instances: int = 2
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    fixed_iters: bool = True
    convergence_iteration: int = 0
    minimum_iters: Optional[int] = None
    return_best_action: bool = True
    line_search_scale: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])
    line_search_type: Any = "APPROX_WOLFE"
    use_cuda_kernel_line_search: bool = False
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9
    history: int = 7
    epsilon: float = 0.01
    use_cuda_kernel_step_direction: bool = False
    stable_mode: bool = True
    use_cuda_kernel_shared_buffers: bool = False
    initial_step_scale: float = 0.1

    def __post_init__(self) -> None:
        for name in ("num_iters", "inner_iters", "num_problems", "history", "convergence_iteration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.num_iters <= 0 or self.inner_iters <= 0 or self.num_problems <= 0 or self.history <= 0:
            raise ValueError("num_iters, inner_iters, num_problems, and history must be positive")
        if self.num_particles is None:
            self.num_particles = len(self.line_search_scale)
        if isinstance(self.num_particles, bool) or not isinstance(self.num_particles, int) or self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if not self.line_search_scale:
            raise ValueError("line_search_scale must not be empty")
        self.line_search_scale = list(self.line_search_scale)
        for scale in self.line_search_scale:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
                raise ValueError("line_search_scale must contain finite nonnegative values")
        for name in (
            "step_scale", "cost_convergence", "cost_delta_threshold", "cost_relative_threshold",
            "converged_ratio", "line_search_wolfe_c_1", "line_search_wolfe_c_2", "epsilon",
            "initial_step_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_scale < 0 or self.cost_convergence < 0 or self.cost_delta_threshold < 0:
            raise ValueError("step scale and convergence thresholds must be nonnegative")
        if not 0 <= self.cost_relative_threshold < 1:
            raise ValueError("cost_relative_threshold must be in [0, 1)")
        if not 0 <= self.converged_ratio <= 1:
            raise ValueError("converged_ratio must be in [0, 1]")
        if self.epsilon <= 0 or self.initial_step_scale < 0:
            raise ValueError("epsilon must be positive and initial_step_scale nonnegative")
        if not 0 < self.line_search_wolfe_c_1 < self.line_search_wolfe_c_2 < 1:
            raise ValueError("Wolfe constants must satisfy 0 < c_1 < c_2 < 1")
        if self.minimum_iters is not None and (
            isinstance(self.minimum_iters, bool)
            or not isinstance(self.minimum_iters, int)
            or self.minimum_iters < 0
        ):
            raise ValueError("minimum_iters must be a nonnegative integer or None")
        for name in ("fixed_iters", "return_best_action", "fix_terminal_action"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if not self.stable_mode:
            raise ValueError("LBFGS stable_mode must be true")
        if self._num_rollout_instances != 2:
            raise ValueError("LBFGS _num_rollout_instances must be 2")
        # These toggles select CUDA-only implementation details upstream.  A
        # regular PyTorch line search/two-loop solve remains available here.
        self.use_cuda_kernel_line_search = False
        self.use_cuda_kernel_step_direction = False
        self.use_cuda_kernel_shared_buffers = False

    @property
    def portable_line_search(self):
        name = LineSearchType(self.line_search_type).value
        # The production portable L-BFGS core exposes Armijo and strong
        # Wolfe.  V2's weak-Wolfe candidate strategy is available from this
        # package directly; L-BFGS maps it to the stronger safe condition.
        if "wolfe" in name:
            return "strong_wolfe"
        return "armijo"

    def update_niters(self, niters: int):
        if niters <= 0:
            raise ValueError("niters must be positive")
        if niters < self.inner_iters or niters % self.inner_iters:
            raise ValueError("num_iters must be a positive multiple of inner_iters")
        self.num_iters = niters


class LBFGSOpt(PortableOptimizer):
    """Batched eager L-BFGS with device-resident two-loop histories.

    This does not emulate CUDA graph capture or the raw CUDA line-search ABI.
    It does retain the solver's useful lifecycle: per-problem curvature
    filtering, fixed candidate scales, best-action selection, warm starts,
    reset/shift/resize, and ordinary CPU/MPS autograd.
    """

    strategy = "lbfgs"

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        if len(rollout_list) != config.num_rollout_instances:
            raise ValueError(
                f"num_rollout_instances {config.num_rollout_instances} "
                f"!= len(rollout_list) {len(rollout_list)}"
            )
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._s_history: torch.Tensor | None = None
        self._y_history: torch.Tensor | None = None
        self._rho_history: torch.Tensor | None = None
        self._reference_action: torch.Tensor | None = None
        self._reference_gradient: torch.Tensor | None = None
        self._pending_action: torch.Tensor | None = None
        self._best_action: torch.Tensor | None = None
        self._best_cost: torch.Tensor | None = None
        self._convergence_count: torch.Tensor | None = None
        self._converged: torch.Tensor | None = None
        self._iteration = 0
        self._original_num_iters = config.num_iters

    @property
    def horizon(self) -> int:
        """Action horizon exposed by the configured rollout."""
        return int(
            getattr(
                self.rollout_fn,
                "horizon",
                getattr(self.rollout_fn, "action_horizon", self.config.action_horizon),
            )
        )

    @property
    def solver_names(self) -> list[str]:
        return [self.config.solver_name]

    def get_all_rollout_instances(self) -> list[Any]:
        return [self.rollout_fn]

    def reset_seed(self) -> bool:
        """Portable L-BFGS is deterministic and owns no random seed state."""
        return True

    def update_niters(self, niters: int) -> None:
        self.config.update_niters(niters)

    def compute_metrics(self, action: torch.Tensor) -> Any:
        metrics = getattr(self.rollout_fn, "compute_metrics", None)
        if callable(metrics):
            return metrics(action)
        return {"cost": self._cost(self._canonical_action(action))}

    # -- limited-memory two-loop state -------------------------------------

    def _clear_history(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self._s_history = self._y_history = self._rho_history = None
            self._reference_action = self._reference_gradient = None
            return
        for value in (self._s_history, self._y_history, self._rho_history,
                      self._reference_action, self._reference_gradient):
            if value is not None and value.shape[0] == mask.shape[0]:
                value[mask] = 0

    def _shift_history(self, shift_steps: int) -> None:
        if shift_steps <= 0:
            return
        shift = int(shift_steps) * self.action_dim
        for value in (self._s_history, self._y_history, self._reference_action, self._reference_gradient):
            if value is not None and shift < value.shape[-1]:
                value.copy_(torch.roll(value, -shift, dims=-1))
                value[..., -shift:] = 0
            elif value is not None:
                value.zero_()
        if self._rho_history is not None:
            self._rho_history.zero_()

    def _record_pair(self, action: torch.Tensor, gradient: torch.Tensor) -> None:
        current_action = action.detach().reshape(action.shape[0], -1)
        current_gradient = gradient.detach().reshape(gradient.shape[0], -1)
        if (
            self._reference_action is None or self._reference_gradient is None
            or self._reference_action.shape != current_action.shape
        ):
            self._reference_action, self._reference_gradient = current_action.clone(), current_gradient.clone()
            return
        s = current_action - self._reference_action
        y = current_gradient - self._reference_gradient
        curvature = (s * y).sum(-1)
        moved = torch.linalg.vector_norm(s, dim=-1) > self.config.epsilon
        valid = torch.isfinite(curvature) & moved & (curvature > self.config.epsilon)
        safe_s = torch.where(valid[:, None], s, torch.zeros_like(s))
        safe_y = torch.where(valid[:, None], y, torch.zeros_like(y))
        rho = torch.where(valid, curvature.reciprocal(), torch.zeros_like(curvature))
        if self._s_history is None or self._s_history.shape[0] != action.shape[0]:
            self._s_history = safe_s[:, None]
            self._y_history = safe_y[:, None]
            self._rho_history = rho[:, None]
        else:
            self._s_history = torch.cat((safe_s[:, None], self._s_history), dim=1)[:, : self.config.history]
            self._y_history = torch.cat((safe_y[:, None], self._y_history), dim=1)[:, : self.config.history]
            self._rho_history = torch.cat((rho[:, None], self._rho_history), dim=1)[:, : self.config.history]
        self._reference_action, self._reference_gradient = current_action.clone(), current_gradient.clone()

    def _two_loop(self, gradient: torch.Tensor) -> torch.Tensor:
        flat = gradient.detach().reshape(gradient.shape[0], -1)
        if self._s_history is None or self._s_history.shape[1] == 0:
            return -flat.reshape_as(gradient)
        assert self._y_history is not None and self._rho_history is not None
        q = flat.clone()
        alpha: list[torch.Tensor] = []
        for index in range(self._s_history.shape[1]):
            current = self._rho_history[:, index] * (self._s_history[:, index] * q).sum(-1)
            alpha.append(current)
            q = q - current[:, None] * self._y_history[:, index]
        gamma = torch.ones_like(self._rho_history[:, 0])
        selected = torch.zeros_like(gamma, dtype=torch.bool)
        for index in range(self._s_history.shape[1]):
            sy = (self._s_history[:, index] * self._y_history[:, index]).sum(-1)
            yy = self._y_history[:, index].square().sum(-1)
            valid = (~selected) & (self._rho_history[:, index] > 0) & (yy > self.config.epsilon)
            gamma = torch.where(valid, sy / yy.clamp_min(self.config.epsilon), gamma)
            selected |= valid
        q = q * gamma[:, None]
        for index in reversed(range(self._s_history.shape[1])):
            beta = self._rho_history[:, index] * (self._y_history[:, index] * q).sum(-1)
            q = q + self._s_history[:, index] * (alpha[index] - beta)[:, None]
        return -q.reshape_as(gradient)

    def _get_step_direction_impl(self, action: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        self._record_pair(action, gradient)
        direction = self._two_loop(gradient)
        max_step = self.action_horizon_step_max
        if max_step is None:
            max_step = self.action_step_max
        if self.config.step_scale not in (0.0, 1.0) and max_step is not None:
            bound = torch.as_tensor(max_step, device=direction.device, dtype=direction.dtype)
            if bool((bound <= 0).any()):
                raise ValueError("action_step_max entries must be positive")
            ratio = (direction.abs() / bound).reshape(direction.shape[0], -1).amax(-1)
            direction = direction / ratio.clamp_min(1.0).reshape((-1,) + (1,) * (direction.ndim - 1))
        if self.config.fix_terminal_action and direction.ndim >= 3:
            direction = direction.clone()
            direction[:, -1] = 0
        return direction

    def _canonical_action(self, action: torch.Tensor) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            raise TypeError("seed_action must be a torch.Tensor")
        batch, horizon, dimension = self.config.num_problems, self.action_horizon, self.action_dim
        # Direct V2 optimizer use commonly supplies a multi-problem seed
        # before an explicit ``update_num_problems`` call.  Preserve that
        # useful convenience only from the default single-problem state.
        if action.ndim == 3 and tuple(action.shape[1:]) == (horizon, dimension) and batch == 1:
            if action.shape[0] != 1:
                self.update_num_problems(int(action.shape[0]))
            return action
        if action.ndim == 3 and tuple(action.shape) == (batch, horizon, dimension):
            return action
        if action.ndim == 2 and tuple(action.shape) == (batch, horizon * dimension):
            return action.reshape(batch, horizon, dimension)
        if batch == 1 and action.ndim == 2 and tuple(action.shape) == (horizon, dimension):
            return action.unsqueeze(0)
        raise ValueError(f"seed_action must have shape [{batch}, {horizon}, {dimension}] or [{batch}, {horizon * dimension}]")

    def _project(self, action: torch.Tensor) -> torch.Tensor:
        low, high = self.action_bound_lows, self.action_bound_highs
        if low is None or high is None:
            return action
        low = torch.as_tensor(low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(high, device=action.device, dtype=action.dtype)
        if bool((low > high).any()):
            raise ValueError("action_bound_lows must not exceed action_bound_highs")
        try:
            return torch.maximum(torch.minimum(action, high), low)
        except RuntimeError as error:
            raise ValueError("action bounds must broadcast to [B, H, D]") from error

    def _cost(self, action: torch.Tensor) -> torch.Tensor:
        value = _objective(self.rollout_fn)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        if value.ndim == 0:
            return value.expand(action.shape[0])
        if value.shape[0] != action.shape[0]:
            raise ValueError("rollout objective must return one value per problem or a scalar")
        return value.reshape(action.shape[0], -1).sum(-1)

    def _gradient(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            leaf = action.detach().requires_grad_(True)
            cost = self._cost(leaf)
            finite = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost)).sum()
            gradient = torch.autograd.grad(finite, leaf, allow_unused=True)[0] if finite.requires_grad else None
        return cost.detach(), torch.zeros_like(action) if gradient is None else gradient.detach()

    def _candidate_step(self, action: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales = torch.as_tensor([0.0, *self.config.line_search_scale], device=action.device, dtype=action.dtype)
        candidates = self._project(action[:, None] + scales.reshape((1, -1) + (1,) * (action.ndim - 1)) * direction[:, None])
        costs = self._cost(candidates.reshape((-1,) + action.shape[1:])).reshape(action.shape[0], -1)
        index = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf)).argmin(-1)
        gather = index.reshape((-1, 1) + (1,) * (action.ndim - 1)).expand((-1, 1) + action.shape[1:])
        return candidates.gather(1, gather).squeeze(1).detach(), costs.gather(1, index[:, None]).squeeze(1).detach()

    def _should_stop(self, previous: torch.Tensor, current: torch.Tensor, iteration: int) -> bool:
        if self.config.fixed_iters:
            return False
        delta = (previous - current).abs()
        absolute = delta <= max(self.config.cost_convergence, self.config.cost_delta_threshold)
        relative = delta <= self.config.cost_relative_threshold * previous.abs().clamp_min(torch.finfo(current.dtype).eps) if self.config.cost_relative_threshold else torch.zeros_like(absolute)
        stable = torch.isfinite(previous) & torch.isfinite(current) & (absolute | relative)
        if self._convergence_count is None or self._convergence_count.shape != stable.shape:
            self._convergence_count = torch.zeros_like(current, dtype=torch.long)
        self._convergence_count = torch.where(stable, self._convergence_count + 1, torch.zeros_like(self._convergence_count))
        self._converged = self._convergence_count >= max(1, self.config.convergence_iteration + 1)
        if self.config.minimum_iters is not None and iteration < self.config.minimum_iters:
            return False
        return bool(self._converged.to(torch.float32).mean().item() >= self.config.converged_ratio)

    def optimize(self, seed_action):
        if not self.enabled:
            return seed_action
        supplied = self._project(self._canonical_action(seed_action)).detach()
        if self._pending_action is not None:
            if self._pending_action.shape != supplied.shape or self._pending_action.device != supplied.device or self._pending_action.dtype != supplied.dtype:
                raise ValueError("reinitialized action must match the next seed shape, dtype, and device")
            x, self._pending_action = self._pending_action, None
        else:
            x = supplied
        self._clear_history()
        best, best_cost = x.clone(), self._cost(x)
        previous = best_cost
        self._convergence_count = self._converged = None
        self._iteration = 0
        trace: list[torch.Tensor] = []
        import time
        start = time.perf_counter()
        for iteration in range(self.config.num_iters):
            _cost, gradient = self._gradient(x)
            direction = self._get_step_direction_impl(x, gradient)
            x, cost = self._candidate_step(x, direction)
            improved = torch.isfinite(cost) & (cost < best_cost)
            best = torch.where(improved.reshape((-1,) + (1,) * (x.ndim - 1)), x, best)
            best_cost = torch.where(improved, cost, best_cost)
            if self.config.store_debug:
                trace.append(best_cost.clone())
            self._iteration += 1
            if self._should_stop(previous, best_cost, iteration + 1):
                break
            previous = best_cost
        if x.device.type == "mps":
            torch.mps.synchronize()
        self.opt_dt = time.perf_counter() - start
        self._best_action, self._best_cost = best.clone(), best_cost.clone()
        self.debug = {"objective": tuple(trace), "converged": None if self._converged is None else self._converged.clone()} if self.config.store_debug else None
        finite = torch.isfinite(self._cost(x)).reshape(x.shape[0], -1).all(-1)
        return best if self.config.return_best_action else torch.where(finite.reshape((-1,) + (1,) * (x.ndim - 1)), x, best)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        canonical = self._project(self._canonical_action(action)).detach().clone()
        checked = None if mask is None else torch.as_tensor(mask, device=canonical.device, dtype=torch.bool)
        if checked is not None and checked.shape != (canonical.shape[0],):
            raise ValueError(f"mask must have shape [{canonical.shape[0]}]")
        super().reinitialize(canonical, checked, clear_optimizer_state, reset_num_iters)
        if reset_num_iters:
            self.config.update_niters(self._original_num_iters)
        if checked is None or self._pending_action is None or self._pending_action.shape != canonical.shape:
            self._pending_action = canonical
        else:
            self._pending_action = torch.where(checked.reshape((-1,) + (1,) * (canonical.ndim - 1)), canonical, self._pending_action)
        self._clear_history(checked)
        self._best_action = self._best_cost = self._converged = self._convergence_count = None
        self._iteration = 0

    def reset(self):
        super().reset(); self._clear_history(); self._pending_action = None
        self._best_action = self._best_cost = self._converged = self._convergence_count = None; self._iteration = 0; self.debug = None

    def shift(self, shift_steps=0):
        if shift_steps < 0: raise ValueError("shift_steps must be nonnegative")
        self._shift_history(int(shift_steps)); self._pending_action = None
        self._best_action = self._best_cost = self._converged = self._convergence_count = None
        return True

    _shift = shift

    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        for rollout in self._rollout_list:
            callback = getattr(rollout, "update_batch_size", None)
            if callable(callback): callback(batch_size=num_problems * self.config.num_particles)
        self._clear_history(); self._pending_action = None; self._best_action = self._best_cost = None

    def get_recorded_trace(self): return self.debug if self.debug is not None else {"debug": [], "debug_cost": []}
    def update_solver_params(self, solver_params):
        if self.config.solver_name not in solver_params: raise ValueError(f"Optimizer {self.config.solver_name} not found in {solver_params}")
        values = solver_params[self.config.solver_name]
        if not isinstance(values, dict): raise TypeError("solver parameters must be a mapping")
        unknown = [key for key in values if not hasattr(self.config, key)]
        if unknown: raise ValueError("unknown optimizer parameter(s): " + ", ".join(sorted(unknown)))
        old = {key: getattr(self.config, key) for key in values}
        try:
            for key, value in values.items(): setattr(self.config, key, value)
            self.config.__post_init__()
        except Exception:
            for key, value in old.items(): setattr(self.config, key, value)
            self.config.__post_init__()
            raise
        return True

    def reset_shape(self):
        for rollout in self._rollout_list:
            callback = getattr(rollout, "reset_shape", None)
            if callable(callback): callback()
        self._clear_history()

    def reset_cuda_graph(self): return None
    def debug_dump(self, file_path=""): del file_path; return None


__all__ = ["LBFGSOptCfg", "LBFGSOpt"]
