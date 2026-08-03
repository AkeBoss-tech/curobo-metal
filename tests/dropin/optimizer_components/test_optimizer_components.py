from __future__ import annotations

import pytest
import torch

from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.best_tracker import BestTracker
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg


def test_action_bounds_preserve_v2_flat_buffers_and_refresh_mutable_values() -> None:
    bounds = ActionBounds(torch.tensor([-2.0, 1.0]), torch.tensor([3.0, 5.0]), 3, 0.25)
    torch.testing.assert_close(bounds.horizon_lows, torch.tensor([-2.0, 1.0, -2.0, 1.0, -2.0, 1.0]))
    torch.testing.assert_close(bounds.horizon_highs, torch.tensor([3.0, 5.0, 3.0, 5.0, 3.0, 5.0]))
    torch.testing.assert_close(bounds.step_max, torch.tensor([1.25, 1.0]))
    assert bounds.action_horizon_bounds_lows.shape == (3, 2)
    bounds.refresh(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 3.0]), 2)
    torch.testing.assert_close(bounds.horizon_step_max, torch.tensor([0.5, 1.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    "lows, highs, horizon",
    [
        (torch.tensor([0]), torch.tensor([1]), 1),
        (torch.tensor([0.0]), torch.tensor([1.0, 2.0]), 1),
        (torch.tensor([2.0]), torch.tensor([1.0]), 1),
        (torch.tensor([0.0]), torch.tensor([1.0]), 0),
    ],
)
def test_action_bounds_reject_invalid_shapes_and_limits(lows, highs, horizon) -> None:
    with pytest.raises((TypeError, ValueError)):
        ActionBounds(lows, highs, horizon)


def test_best_tracker_tracks_strict_improvement_and_masked_clear() -> None:
    tracker = BestTracker(DeviceCfg())
    tracker.resize(2, 2, 1)
    state = OptimizationIterationState(
        action=torch.tensor([[[3.0], [4.0]], [[7.0], [8.0]]]),
        cost=torch.tensor([5.0, 2.0]),
    )
    tracker.update(state, 2, 1, 1e-4, 1e-4, 1)
    torch.testing.assert_close(state.best_cost, torch.tensor([5.0, 2.0]))
    assert state.current_iteration.tolist() == [1, 1]
    assert state.converged.tolist() == [0, 0]

    state.action = torch.tensor([[[9.0], [10.0]], [[11.0], [12.0]]])
    state.cost = torch.tensor([5.0, 1.0])  # equality must not replace the first best action
    tracker.update(state, 2, 1, 1e-4, 1e-4, 1)
    torch.testing.assert_close(state.best_cost, torch.tensor([5.0, 1.0]))
    torch.testing.assert_close(state.best_action[0], torch.tensor([[3.0], [4.0]]))
    torch.testing.assert_close(state.best_action[1], torch.tensor([[11.0], [12.0]]))
    assert state.converged.tolist() == [1, 0]
    assert BestTracker.check_convergence(state.converged, 0.5) is False
    assert BestTracker.check_convergence(state.converged, 0.0) is True

    tracker.clear(torch.tensor([False, True]))
    assert tracker.cost[0].item() == pytest.approx(5.0)
    assert tracker.cost[1].item() > 1e12
    assert tracker.current_iteration.tolist() == [2, 0]


def test_debug_recorder_returns_detached_action_and_cost_history() -> None:
    action = torch.ones((2, 2, 3), requires_grad=True)
    cost = action.square().sum(dim=(-1, -2))
    recorder = DebugRecorder()
    recorder.record(OptimizationIterationState(action=action, cost=cost), 2, 3)
    trace = recorder.get_trace()
    assert trace.keys() == ("debug", "debug_cost")
    assert len(trace) == len(trace["debug"]) == len(trace["debug_cost"]) == 1
    assert not trace["debug"][0].requires_grad and not trace["debug_cost"][0].requires_grad
    action.data.fill_(99.0)
    assert trace["debug"][0][0, 0, 0].item() == 1.0
    assert recorder.trace[0].action.shape == (2, 2, 3)
    recorder.clear()
    assert len(recorder.get_trace()) == 0
    assert recorder.get_trace()["debug"] == recorder.get_trace()["debug_cost"] == []


def test_optimizer_components_mps_without_cpu_fallback() -> None:
    if not torch.backends.mps.is_available():
        return
    device_cfg = DeviceCfg(device="mps", dtype=torch.float32)
    tracker = BestTracker(device_cfg)
    tracker.resize(1, 2, 2)
    state = OptimizationIterationState(
        action=torch.ones((1, 2, 2), device="mps"), cost=torch.tensor([1.0], device="mps")
    )
    tracker.update(state, 2, 2, 0.0, 0.0, 0)
    assert tracker.action.device.type == state.best_action.device.type == "mps"
    bounds = ActionBounds(torch.full((2,), -1.0, device="mps"), torch.ones(2, device="mps"), 2)
    assert bounds.horizon_step_max.device.type == "mps"
