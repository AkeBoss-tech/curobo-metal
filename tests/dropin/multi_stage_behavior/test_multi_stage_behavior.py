"""Direct CPU/MPS behaviour coverage for portable MultiStageOptimizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.types.device_cfg import DeviceCfg


class _Rollout:
    action_horizon = 2
    action_dim = 2

    def __init__(self):
        self.reset_shape_calls = 0
        self.reset_seed_calls = 0

    def reset_shape(self):
        self.reset_shape_calls += 1

    def reset_seed(self):
        self.reset_seed_calls += 1


class _Stage:
    def __init__(self, name: str, offset: float, rollout: _Rollout, *, enabled: bool = True):
        self.config = SimpleNamespace(
            solver_name=name, num_problems=2, device_cfg=DeviceCfg(), value=0
        )
        self.action_horizon = 2
        self.action_dim = 2
        self._rollout_list = [rollout]
        self._enabled = enabled
        self.offset = offset
        self.calls: list[tuple[str, object]] = []

    @property
    def enabled(self):
        return self._enabled

    def optimize(self, action):
        self.calls.append(("optimize", tuple(action.shape)))
        return action + self.offset

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        self.calls.append(("reinitialize", (action.clone(), mask, clear_optimizer_state, reset_num_iters)))

    def _shift(self, steps):
        self.calls.append(("shift", steps))
        return self.offset >= 0

    def update_num_problems(self, problems):
        self.config.num_problems = problems
        self.calls.append(("num_problems", problems))

    def update_rollout_params(self, goal):
        self.calls.append(("goal", goal))

    def update_goal_dt(self, dt):
        self.calls.append(("dt", dt))

    def reset(self):
        self.calls.append(("reset", None))

    def reset_shape(self):
        self.calls.append(("reset_shape", None))

    def reset_seed(self):
        self.calls.append(("reset_seed", None))
        return True

    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA Graph capture is unavailable")

    def get_all_rollout_instances(self):
        return self._rollout_list

    def get_recorded_trace(self):
        return {"debug": [self.config.solver_name], "debug_cost": [self.offset]}

    def update_niters(self, niters):
        self.calls.append(("niters", niters))

    def update_solver_params(self, values):
        self.calls.append(("params", values))
        self.config.value = values[self.config.solver_name]["value"]
        return True

    def debug_dump(self, file_path=""):
        self.calls.append(("dump", file_path))
        return self.config.solver_name


def _stages(*, second_enabled=True):
    rollout = _Rollout()
    return rollout, _Stage("warm", 1.0, rollout), _Stage("refine", 2.0, rollout, enabled=second_enabled)


def test_stage_selection_seed_layout_and_trace_lifecycle():
    rollout, first, second = _stages()
    optimizer = MultiStageOptimizer([first, second])
    flattened = torch.zeros((2, 4))

    output = optimizer.optimize(flattened)

    torch.testing.assert_close(output, torch.full((2, 2, 2), 3.0))
    assert first.calls[0] == ("optimize", (2, 2, 2))
    assert second.calls[0] == ("optimize", (2, 2, 2))
    assert len(optimizer.last_stage_outputs) == 2
    assert optimizer._last_iteration_state.best_action is output
    assert optimizer.solve_time >= 0.0
    assert optimizer.get_all_rollout_instances() == [rollout, rollout]
    assert optimizer.get_recorded_trace() == {
        "debug": ["warm", "refine"], "debug_cost": [1.0, 2.0]
    }
    assert optimizer.debug_dump("trace.pt") == ["warm", "refine"]
    assert [call for call in first.calls if call[0] == "dump"] == [("dump", "trace.pt")]


def test_disabled_wrapper_and_disabled_stage_preserve_correct_seed():
    _, first, second = _stages(second_enabled=False)
    optimizer = MultiStageOptimizer([first, second])
    seed = torch.zeros((2, 2, 2))
    torch.testing.assert_close(optimizer.optimize(seed), torch.ones_like(seed))
    assert not any(name == "optimize" for name, _ in second.calls)

    optimizer.disable()
    result = optimizer.optimize(seed)
    torch.testing.assert_close(result, seed)
    assert optimizer.last_stage_outputs == ()


def test_lifecycle_broadcasts_without_short_circuiting_and_validates_params():
    rollout, first, second = _stages()
    first.offset = -1.0  # makes _shift report false, but refine still must see it.
    optimizer = MultiStageOptimizer([first, second])
    action = torch.zeros((2, 2, 2))
    mask = torch.tensor([True, False])
    optimizer.reinitialize(action, mask=mask, clear_optimizer_state=False, reset_num_iters=True)
    assert optimizer.shift(1) is False
    assert ("shift", 1) in first.calls and ("shift", 1) in second.calls
    optimizer.update_num_problems(3)
    assert first.config.num_problems == second.config.num_problems == optimizer.config.num_problems == 3
    optimizer.update_rollout_params("goal")
    optimizer.update_goal_dt(0.1)
    optimizer.update_niters(7)
    optimizer.reset_shape()
    optimizer.reset_seed()
    optimizer.reset_cuda_graph()
    assert rollout.reset_shape_calls == rollout.reset_seed_calls == 1
    assert optimizer.update_solver_params({"warm": {"value": 4}, "refine": {"value": 9}})
    assert (first.config.value, second.config.value) == (4, 9)
    with pytest.raises(ValueError, match="not found"):
        optimizer.update_solver_params({"missing": {"value": 1}})
    with pytest.raises(RuntimeError, match="ambiguous"):
        optimizer.compute_metrics(action)


def test_invalid_stage_action_size_and_seed_shape_are_rejected():
    _, first, second = _stages()
    second.action_dim = 3
    with pytest.raises(ValueError, match="same action size"):
        MultiStageOptimizer([first, second])

    _, first, second = _stages()
    optimizer = MultiStageOptimizer([first, second])
    with pytest.raises(ValueError, match="seed_action"):
        optimizer.optimize(torch.zeros((4, 2)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_preserves_device_and_runs_stage_chain_without_fallback():
    _, first, second = _stages()
    optimizer = MultiStageOptimizer([first, second])
    seed = torch.zeros((2, 2, 2), device="mps")
    result = optimizer.optimize(seed)
    assert result.device.type == "mps"
    torch.testing.assert_close(result.cpu(), torch.full((2, 2, 2), 3.0))
