"""Limited-memory SR1 optimizer implemented with ordinary PyTorch tensors."""

from __future__ import annotations

import torch

from curobo._src.optim._portable import _objective
from .lbfgs import LBFGSOpt


def jit_lsr1_compute_step_direction(
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad: torch.Tensor,
    m: int,
    epsilon: float,
    stable_mode: bool,
    hessian_0: torch.Tensor,
) -> torch.Tensor:
    """Apply the limited-memory SR1 inverse-Hessian update to ``grad``.

    Histories are ordered newest-first.  Updates whose denominator is too
    small are ignored, which is the stable portable counterpart to V2's raw
    kernel path and prevents non-finite steps on MPS.
    """

    if y_buffer.shape != s_buffer.shape:
        raise ValueError("y_buffer and s_buffer must have matching shape")
    if grad.shape[0] != y_buffer.shape[0] or grad.shape[-1] != y_buffer.shape[-1]:
        raise ValueError("history and gradient dimensions must agree")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not stable_mode:
        raise ValueError("LSR1 stable_mode must be true")
    history = min(int(m), y_buffer.shape[-2])
    output = hessian_0 * grad
    for index in range(history):
        y = y_buffer[:, index]
        s = s_buffer[:, index]
        # hessian_0 has batch-compatible [B, 1, 1] shape in V2; broadcasting
        # also supports [B, 1] or scalar values for direct callers.
        h0_y = hessian_0.reshape(hessian_0.shape[0], -1)[:, :1] * y
        u = s - h0_y
        denominator = (u * y).sum(-1, keepdim=True)
        numerator = (u * grad.reshape(grad.shape[0], -1)).sum(-1, keepdim=True)
        valid = denominator.abs() > epsilon
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        update = torch.where(valid, u * (numerator / safe_denominator), torch.zeros_like(u))
        output = output + update.reshape_as(output)
    return -output


class LSR1Opt(LBFGSOpt):
    """Batched L-SR1 with a deterministic, no-worse fixed-candidate step."""

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        self._s_history: torch.Tensor | None = None
        self._y_history: torch.Tensor | None = None

    def _cost(self, action: torch.Tensor) -> torch.Tensor:
        value = _objective(self.rollout_fn)(action)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=action.device, dtype=action.dtype)
        return value.reshape(action.shape[0], -1).sum(-1)

    def _candidate_step(self, action: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales = torch.as_tensor(
            [0.0, *self.config.line_search_scale], device=action.device, dtype=action.dtype
        )
        candidates = action[:, None] + scales.reshape(1, -1, *([1] * (action.ndim - 1))) * direction[:, None]
        flat = candidates.reshape((-1,) + tuple(action.shape[1:]))
        costs = self._cost(flat).reshape(action.shape[0], scales.numel())
        index = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf)).argmin(-1)
        gather = index.reshape(-1, 1, *([1] * (action.ndim - 1))).expand(-1, 1, *action.shape[1:])
        return candidates.gather(1, gather).squeeze(1).detach(), costs.gather(1, index[:, None]).squeeze(1)

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return seed_action
        if not isinstance(seed_action, torch.Tensor) or seed_action.ndim < 3:
            raise ValueError("LSR1Opt expects [batch, horizon, action_dim] seed_action")
        x = seed_action.detach()
        batch, dimensions = x.shape[0], x[0].numel()
        s_history = torch.zeros(batch, 0, dimensions, dtype=x.dtype, device=x.device)
        y_history = torch.zeros_like(s_history)
        best, best_cost = x.clone(), self._cost(x)
        trace: list[torch.Tensor] = []
        for _ in range(self.config.num_iters):
            leaf = x.detach().requires_grad_(True)
            cost = self._cost(leaf)
            gradient = torch.autograd.grad(
                torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost)).sum(),
                leaf,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(leaf)
            hessian_0 = torch.ones(batch, 1, 1, dtype=x.dtype, device=x.device)
            direction = jit_lsr1_compute_step_direction(
                y_history, s_history, gradient.reshape(batch, 1, dimensions),
                self.config.history, self.config.epsilon, self.config.stable_mode, hessian_0,
            ).reshape_as(x)
            x_next, candidate_cost = self._candidate_step(leaf.detach(), direction.detach())
            next_leaf = x_next.detach().requires_grad_(True)
            next_cost = self._cost(next_leaf)
            next_gradient = torch.autograd.grad(
                torch.where(torch.isfinite(next_cost), next_cost, torch.zeros_like(next_cost)).sum(),
                next_leaf,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )[0]
            if next_gradient is None:
                next_gradient = torch.zeros_like(next_leaf)
            s = (x_next - leaf.detach()).reshape(batch, dimensions)
            y = (next_gradient.detach() - gradient.detach()).reshape(batch, dimensions)
            # Newest-first fixed-size history; reject no-op pairs early.
            valid_pair = (s * y).sum(-1).abs() > self.config.epsilon
            if bool(valid_pair.any().item()):
                s = torch.where(valid_pair[:, None], s, torch.zeros_like(s))
                y = torch.where(valid_pair[:, None], y, torch.zeros_like(y))
                s_history = torch.cat((s[:, None], s_history), dim=1)[:, : self.config.history]
                y_history = torch.cat((y[:, None], y_history), dim=1)[:, : self.config.history]
            x = x_next
            improved = torch.isfinite(candidate_cost) & (candidate_cost < best_cost)
            best = torch.where(improved[:, None, None], x, best)
            best_cost = torch.where(improved, candidate_cost, best_cost)
            if self.config.store_debug:
                trace.append(best_cost.detach().clone())
        self._s_history, self._y_history = s_history.detach(), y_history.detach()
        self.debug = {"objective": tuple(trace)} if self.config.store_debug else None
        return best if self.config.return_best_action else x

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        super().reinitialize(action, mask, clear_optimizer_state, reset_num_iters)
        if clear_optimizer_state:
            self._s_history = None
            self._y_history = None


__all__ = ["LSR1Opt", "jit_lsr1_compute_step_direction"]
