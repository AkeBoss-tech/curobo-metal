"""Portable, batched line-search strategies for the pinned V2 API.

The upstream implementation has a CUDA kernel fast path, but its ordinary
PyTorch path is a useful contract in its own right: evaluate a fixed set of
candidate steps per problem, deterministically choose the largest acceptable
step for Armijo/Wolfe variants, and keep all values on the input device.  This
module implements that contract without a CUDA graph or a raw CUDA kernel.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import torch

from .line_search_result import LineSearchResult
from .line_search_state import LineSearchState


class LineSearchType(Enum):
    """Pinned V2 line-search names (with backwards-compatible enum parsing)."""

    GREEDY = "greedy"
    ARMIJO = "armijo"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    APPROX_WOLFE = "approx_wolfe"
    APPROX_STRONG_WOLFE = "approx_strong_wolfe"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            normal = value.strip().lower()
            if normal.startswith("linesearchtype."):
                normal = normal.rsplit(".", 1)[-1]
            for member in cls:
                if normal in {member.name.lower(), member.value}:
                    return member
        return None


def _cost_matrix(cost: torch.Tensor, batch: int, candidates: int) -> torch.Tensor:
    """Normalize rollout costs to one scalar per ``(problem, candidate)``.

    Rollouts in the V2 stack may return a scalar, a horizon of costs, or a
    trailing metric vector.  Only the latter dimensions are reduced; batch and
    candidate dimensions retain their independent selection semantics.
    """

    if not isinstance(cost, torch.Tensor):
        raise TypeError("line-search cost must be a torch.Tensor")
    if cost.ndim == 0:
        return cost.expand(batch, candidates)
    if cost.shape[:2] == (batch, candidates):
        return cost.reshape(batch, candidates, -1).sum(-1)
    if cost.shape[0] == batch * candidates:
        return cost.reshape(batch, candidates, -1).sum(-1)
    if batch == 1 and cost.shape[0] == candidates:
        return cost.reshape(1, candidates, -1).sum(-1)
    raise ValueError(
        "line-search cost must start with [batch, candidates] or flatten "
        f"those dimensions; got {tuple(cost.shape)}, expected ({batch}, {candidates}, ...)"
    )


def _gradient_tensor(
    gradient: torch.Tensor, points: torch.Tensor, batch: int, candidates: int
) -> torch.Tensor:
    """Normalize gradients to the same rank as candidate points."""

    if not isinstance(gradient, torch.Tensor):
        raise TypeError("line-search gradient must be a torch.Tensor")
    if gradient.shape == points.shape:
        return gradient
    if gradient.shape[0] == batch * candidates and gradient.shape[1:] == points.shape[2:]:
        return gradient.reshape_as(points)
    raise ValueError(
        "line-search gradient must have candidate-point shape or flattened "
        f"[batch*candidates, ...]; got {tuple(gradient.shape)}, expected {tuple(points.shape)}"
    )


def _index_state(
    points: torch.Tensor, costs: torch.Tensor, gradients: torch.Tensor, idx: torch.Tensor
) -> LineSearchState:
    """Gather per-problem candidates with stable first-index tie handling."""

    event_dims = points.ndim - 2
    gather = idx.reshape(idx.shape + (1,) * event_dims).unsqueeze(1)
    gather = gather.expand((points.shape[0], 1) + tuple(points.shape[2:]))
    action = points.gather(1, gather).squeeze(1).detach()
    gradient = gradients.gather(1, gather).squeeze(1).detach()
    cost = costs.gather(1, idx.unsqueeze(1)).squeeze(1).detach()
    return LineSearchState(action=action, cost=cost, gradient=gradient, idxs=idx.detach())


class LineSearchStrategy:
    """Base fixed-candidate line search.

    ``search`` returns :class:`LineSearchResult`, as did the initial portable
    facade.  The result is deliberately separate from CUDA graph iteration
    buffers; callers can copy it into an ``OptimizationIterationState`` when
    they own one.
    """

    def update_num_problems(self, num_problems: int, context: Any | None = None):
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        self.num_problems = int(num_problems)
        if context is not None:
            context.update_num_problems(num_problems)
        return self

    @staticmethod
    def jit_get_x_set(
        step_vec: torch.Tensor, x: torch.Tensor, line_search_scales: torch.Tensor
    ) -> torch.Tensor:
        """Return candidates shaped ``[B, N, *event]`` without host copies."""

        if x.shape != step_vec.shape or x.ndim < 2:
            raise ValueError("x and step_vec must have matching [batch, *event] shape")
        scales = torch.as_tensor(line_search_scales, device=x.device, dtype=x.dtype).reshape(-1)
        if scales.numel() == 0:
            raise ValueError("line_search_scales must not be empty")
        return x.unsqueeze(1) + scales.reshape((1, -1) + (1,) * (x.ndim - 1)) * step_vec.unsqueeze(1)

    @staticmethod
    def scale_action(
        dx: torch.Tensor,
        action_step_max: torch.Tensor | float | None,
        step_scale: float,
        fix_terminal_action: bool,
        action_horizon: int,
    ) -> torch.Tensor:
        """Limit a whole action step and optionally fix its terminal action.

        This mirrors V2's *relative* max-step scaling rather than clipping each
        coordinate independently.  ``step_scale`` is a compatibility trigger:
        upstream applies max-step normalization when it differs from 0/1.
        """

        output = dx.clone()
        if step_scale not in (0.0, 1.0) and action_step_max is not None:
            max_step = torch.as_tensor(action_step_max, device=dx.device, dtype=dx.dtype)
            if bool((max_step <= 0).any().item()):
                raise ValueError("action_step_max entries must be positive")
            ratio = (output.abs() / max_step).reshape(output.shape[0], -1).amax(-1)
            output = output / ratio.clamp_min(1.0).reshape((-1,) + (1,) * (output.ndim - 1))
        if fix_terminal_action and action_horizon > 1:
            output[:, action_horizon - 1 :, ...] = 0
        return output

    def _prepare_search_points(self, iteration_state: Any, context: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = getattr(iteration_state, "exploration_action", None)
        if x is None:
            x = iteration_state.action
        direction = iteration_state.step_direction
        if not isinstance(x, torch.Tensor) or not isinstance(direction, torch.Tensor):
            raise TypeError("iteration state must provide tensor action and step_direction")
        if x.shape != direction.shape:
            raise ValueError("action and step_direction must have equal shape")
        direction = direction.detach()
        if (getattr(context, "step_scale", 1.0) not in (0.0, 1.0)) or getattr(context, "fix_terminal_action", False):
            direction = self.scale_action(
                direction,
                getattr(context, "action_horizon_step_max", None),
                getattr(context, "step_scale", 1.0),
                getattr(context, "fix_terminal_action", False),
                getattr(context, "action_horizon", x.shape[1] if x.ndim > 1 else 1),
            )
        scales = torch.as_tensor(context.line_search_scale, device=x.device, dtype=x.dtype).reshape(-1)
        return self.jit_get_x_set(direction, x.detach(), scales), direction, scales

    @staticmethod
    def _call_objective(points: torch.Tensor, context: Any) -> tuple[torch.Tensor, torch.Tensor]:
        result = context.compute_costs_and_gradients(points)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise TypeError("compute_costs_and_gradients must return (cost, gradient)")
        cost, gradient = result[:2]
        batch, candidates = points.shape[:2]
        return _cost_matrix(cost, batch, candidates), _gradient_tensor(gradient, points, batch, candidates)

    def _result_for_indices(
        self, points: torch.Tensor, costs: torch.Tensor, gradients: torch.Tensor, selected: torch.Tensor, exploration: torch.Tensor | None = None
    ) -> LineSearchResult:
        if exploration is None:
            exploration = selected
        return LineSearchResult(
            selected_state=_index_state(points, costs, gradients, selected),
            exploration_state=_index_state(points, costs, gradients, exploration),
        )

    def search(self, iteration_state: Any, context: Any) -> LineSearchResult:
        points, _direction, _scales = self._prepare_search_points(iteration_state, context)
        costs, gradients = self._call_objective(points, context)
        # ``min`` keeps the first candidate on exact ties, matching V2's
        # deterministic tensor reduction behavior.
        selected = torch.where(torch.isfinite(costs), costs, torch.full_like(costs, torch.inf)).argmin(dim=1)
        return self._result_for_indices(points, costs, gradients, selected)


class GreedyLineSearchStrategy(LineSearchStrategy):
    """Choose the finite candidate with minimum cost."""


class ArmijoLineSearchStrategy(LineSearchStrategy):
    """Choose the largest candidate satisfying sufficient decrease."""

    def _acceptable(
        self, iteration_state: Any, context: Any, points: torch.Tensor, direction: torch.Tensor, scales: torch.Tensor, costs: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        base_cost = getattr(iteration_state, "exploration_cost", None)
        if not isinstance(base_cost, torch.Tensor):
            base_cost = costs[:, 0]
        else:
            base_cost = base_cost.reshape(costs.shape[0], -1).sum(-1)
        base_gradient = getattr(iteration_state, "exploration_gradient", None)
        if not isinstance(base_gradient, torch.Tensor):
            base_gradient = gradients[:, 0]
        derivative = (base_gradient.reshape(base_gradient.shape[0], -1) * direction.reshape(direction.shape[0], -1)).sum(-1)
        rhs = base_cost[:, None] + float(getattr(context, "line_search_c_1", 1e-5)) * scales[None, :] * derivative[:, None]
        return torch.isfinite(costs) & (costs <= rhs)

    @staticmethod
    def _largest(mask: torch.Tensor, scales: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        rank = torch.where(mask, scales[None, :], torch.full_like(scales[None, :], -torch.inf))
        any_valid = mask.any(dim=1)
        return torch.where(any_valid, rank.argmax(dim=1), fallback)

    def search(self, iteration_state: Any, context: Any) -> LineSearchResult:
        points, direction, scales = self._prepare_search_points(iteration_state, context)
        costs, gradients = self._call_objective(points, context)
        acceptable = self._acceptable(iteration_state, context, points, direction, scales, costs, gradients)
        selected = self._largest(acceptable, scales, torch.zeros(costs.shape[0], device=costs.device, dtype=torch.long))
        return self._result_for_indices(points, costs, gradients, selected)


class BaseWolfeLineSearchStrategy(ArmijoLineSearchStrategy):
    """PyTorch Wolfe selection shared by weak/strong and approximate forms."""

    def _curvature(self, candidate_directional: torch.Tensor, initial_directional: torch.Tensor, context: Any) -> torch.Tensor:
        raise NotImplementedError

    def _fallback(self, wolfe: torch.Tensor, armijo: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        wolfe_idx = self._largest(wolfe, scales, torch.zeros(wolfe.shape[0], device=wolfe.device, dtype=torch.long))
        return wolfe_idx

    def search(self, iteration_state: Any, context: Any) -> LineSearchResult:
        points, direction, scales = self._prepare_search_points(iteration_state, context)
        costs, gradients = self._call_objective(points, context)
        armijo = self._acceptable(iteration_state, context, points, direction, scales, costs, gradients)
        directional = (gradients.reshape(*gradients.shape[:2], -1) * direction[:, None].reshape(direction.shape[0], 1, -1)).sum(-1)
        base_gradient = getattr(iteration_state, "exploration_gradient", None)
        initial = (
            (base_gradient.reshape(base_gradient.shape[0], -1) * direction.reshape(direction.shape[0], -1)).sum(-1)
            if isinstance(base_gradient, torch.Tensor)
            else directional[:, 0]
        )
        wolfe = armijo & self._curvature(directional, initial[:, None], context)
        selected = self._largest(wolfe, scales, torch.zeros(costs.shape[0], device=costs.device, dtype=torch.long))
        exploration = self._fallback(wolfe, armijo, scales)
        return self._result_for_indices(points, costs, gradients, selected, exploration)


class WolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    def _curvature(self, candidate_directional: torch.Tensor, initial_directional: torch.Tensor, context: Any) -> torch.Tensor:
        return candidate_directional >= float(getattr(context, "line_search_c_2", 0.9)) * initial_directional

    def _fallback(self, wolfe: torch.Tensor, armijo: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        fallback = self._largest(armijo, scales, torch.zeros(wolfe.shape[0], device=wolfe.device, dtype=torch.long))
        selected = self._largest(wolfe, scales, fallback)
        return torch.where(wolfe.any(dim=1), selected, fallback)


class StrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    def _curvature(self, candidate_directional: torch.Tensor, initial_directional: torch.Tensor, context: Any) -> torch.Tensor:
        return candidate_directional.abs() <= float(getattr(context, "line_search_c_2", 0.9)) * initial_directional.abs()


class ApproxWolfeLineSearchStrategy(WolfeLineSearchStrategy):
    def _fallback(self, wolfe: torch.Tensor, armijo: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        fallback = self._largest(armijo, scales, torch.zeros(wolfe.shape[0], device=wolfe.device, dtype=torch.long))
        default = torch.full_like(fallback, min(1, scales.numel() - 1))
        fallback = torch.where(armijo.any(dim=1), fallback, default)
        selected = self._largest(wolfe, scales, fallback)
        return torch.where(wolfe.any(dim=1), selected, fallback)


class ApproxStrongWolfeLineSearchStrategy(StrongWolfeLineSearchStrategy):
    def _fallback(self, wolfe: torch.Tensor, armijo: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return ApproxWolfeLineSearchStrategy._fallback(self, wolfe, armijo, scales)


class LineSearchStrategyFactory:
    _map: dict[LineSearchType, type[LineSearchStrategy]] = {
        LineSearchType.GREEDY: GreedyLineSearchStrategy,
        LineSearchType.ARMIJO: ArmijoLineSearchStrategy,
        LineSearchType.WOLFE: WolfeLineSearchStrategy,
        LineSearchType.STRONG_WOLFE: StrongWolfeLineSearchStrategy,
        LineSearchType.APPROX_WOLFE: ApproxWolfeLineSearchStrategy,
        LineSearchType.APPROX_STRONG_WOLFE: ApproxStrongWolfeLineSearchStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_type: LineSearchType | str) -> LineSearchStrategy:
        try:
            kind = LineSearchType(strategy_type)
        except ValueError as error:
            raise ValueError(f"Unknown line search type: {strategy_type}") from error
        return cls._map[kind]()

    @classmethod
    def register_strategy(cls, strategy_type: LineSearchType | str, strategy: type[LineSearchStrategy] | LineSearchStrategy):
        kind = LineSearchType(strategy_type)
        if kind in cls._map:
            raise ValueError(f"Line search strategy {kind.value} already registered")
        candidate = strategy if isinstance(strategy, type) else type(strategy)
        if not issubclass(candidate, LineSearchStrategy):
            raise TypeError("strategy must inherit LineSearchStrategy")
        cls._map[kind] = candidate


__all__ = [
    "LineSearchType", "LineSearchStrategy", "GreedyLineSearchStrategy",
    "ArmijoLineSearchStrategy", "BaseWolfeLineSearchStrategy",
    "WolfeLineSearchStrategy", "StrongWolfeLineSearchStrategy",
    "ApproxWolfeLineSearchStrategy", "ApproxStrongWolfeLineSearchStrategy",
    "LineSearchStrategyFactory",
]
