"""Lifecycle and device-contract coverage for ``MultiStageOptimizer``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg


class _Stage:
    def __init__(
        self,
        name: str,
        *,
        horizon: int = 2,
        action_dim: int = 2,
        problems: int = 2,
        offset: float = 0.0,
    ) -> None:
        self.config = SimpleNamespace(
            solver_name=name, num_problems=problems, device_cfg=DeviceCfg()
        )
        self.action_horizon = horizon
        self.action_dim = action_dim
        self._rollout_list = [object()]
        self._enabled = True
        self.offset = offset
        self.reinitialized: list[tuple[int, ...]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def optimize(self, action: torch.Tensor) -> torch.Tensor:
        return action + self.offset

    def reinitialize(self, action: torch.Tensor, **_: object) -> None:
        self.reinitialized.append(tuple(action.shape))

    def update_num_problems(self, value: int) -> None:
        self.config.num_problems = value


def test_direct_iteration_state_runs_once_and_uses_best_action() -> None:
    warm = _Stage("warm", horizon=1, action_dim=4, offset=1.0)
    refine = _Stage("refine", horizon=2, action_dim=2, offset=2.0)
    optimizer = MultiStageOptimizer([warm, refine])
    base = torch.zeros((2, 2, 2))
    best = torch.full((2, 2, 2), 4.0)

    result = optimizer._opt_iters(OptimizationIterationState(action=base, best_action=best))

    assert result.action is result.best_action
    assert result.exploration_action is result.action
    torch.testing.assert_close(result.best_action, torch.full((2, 2, 2), 7.0))
    assert [tuple(value.shape) for value in optimizer.last_stage_outputs] == [
        (2, 1, 4),
        (2, 2, 2),
    ]


def test_constructor_rejects_mismatched_problem_counts_early() -> None:
    first = _Stage("first", problems=1)
    final = _Stage("final", problems=2)

    with pytest.raises(ValueError, match="same num_problems"):
        MultiStageOptimizer([first, final])


def test_reinitialize_uses_each_stage_layout_and_preserves_final_layout() -> None:
    warm = _Stage("warm", horizon=1, action_dim=4)
    refine = _Stage("refine", horizon=2, action_dim=2)
    optimizer = MultiStageOptimizer([warm, refine])

    optimizer.reinitialize(torch.zeros((2, 2, 2)))

    assert warm.reinitialized == [(2, 1, 4)]
    assert refine.reinitialized == [(2, 2, 2)]


def test_stage_output_must_preserve_tensor_transport_contract() -> None:
    class _WrongDtypeStage(_Stage):
        def optimize(self, action: torch.Tensor) -> torch.Tensor:
            return action.to(torch.float64)

    optimizer = MultiStageOptimizer([_WrongDtypeStage("wrong")])
    with pytest.raises(ValueError, match="changed action dtype"):
        optimizer.optimize(torch.zeros((2, 2, 2), dtype=torch.float32))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_direct_iteration_state_stays_on_mps_without_fallback() -> None:
    optimizer = MultiStageOptimizer([_Stage("stage", offset=1.0)])
    seed = torch.zeros((2, 2, 2), device="mps")

    result = optimizer._opt_iters(OptimizationIterationState(action=seed))

    assert result.best_action is not None
    assert result.best_action.device.type == "mps"
    torch.testing.assert_close(result.best_action.cpu(), torch.ones((2, 2, 2)))
