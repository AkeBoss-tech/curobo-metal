"""Portable behavior for the pinned rollout protocol surface."""

from __future__ import annotations

import inspect

import pytest
import torch

from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.rollout.rollout_rosenbrock import RosenbrockCfg, RosenbrockRollout
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.types.device_cfg import DeviceCfg


def test_protocol_reexports_canonical_metrics_types():
    from curobo._src.rollout.rollout_protocol import (
        CostCollection as ProtocolCostCollection,
        CostsAndConstraints as ProtocolCostsAndConstraints,
        RolloutMetrics as ProtocolRolloutMetrics,
        RolloutResult as ProtocolRolloutResult,
    )

    assert ProtocolCostCollection is CostCollection
    assert ProtocolCostsAndConstraints is CostsAndConstraints
    assert ProtocolRolloutMetrics is RolloutMetrics
    assert ProtocolRolloutResult is RolloutResult


def test_concrete_portable_rollouts_satisfy_complete_runtime_protocol():
    rosenbrock = RosenbrockRollout()
    robot = RobotRollout()
    assert isinstance(rosenbrock, Rollout)
    assert isinstance(robot, Rollout)

    required = (
        "action_dim", "action_horizon", "action_bound_lows", "action_bound_highs", "dt",
        "sum_horizon", "evaluate_action", "compute_metrics_from_state",
        "compute_metrics_from_action", "update_params", "update_batch_size", "update_dt",
        "reset", "reset_shape", "reset_seed",
    )
    assert all(hasattr(rosenbrock, name) for name in required)
    assert all(hasattr(robot, name) for name in required)


def test_protocol_is_structural_but_rejects_partial_implementations():
    class Incomplete:
        @property
        def action_dim(self) -> int:
            return 2

    assert not isinstance(object(), Rollout)
    assert not isinstance(Incomplete(), Rollout)


def test_protocol_annotations_and_lifecycle_signatures_match_pinned_surface():
    assert Rollout.action_dim.fget.__annotations__["return"] == "int"
    assert Rollout.action_bound_lows.fget.__annotations__["return"] == "torch.Tensor"
    assert Rollout.sum_horizon.fget.__annotations__["return"] == "bool"
    assert str(inspect.signature(Rollout.evaluate_action)) == "(self, act_seq: 'torch.Tensor', **kwargs) -> 'RolloutResult'"
    assert str(inspect.signature(Rollout.reset)) == "(self, reset_problem_ids: 'Optional[torch.Tensor]' = None, **kwargs) -> 'bool'"
    assert str(inspect.signature(Rollout.update_dt)) == "(self, dt: 'Union[float, torch.Tensor]', **kwargs) -> 'bool'"


def test_rosenbrock_protocol_evaluates_and_resets_on_cpu():
    rollout = RosenbrockRollout(RosenbrockCfg(DeviceCfg(), dimensions=3, time_action_horizon=2))
    action = torch.zeros((2, 2, 3), requires_grad=True)
    result = rollout.evaluate_action(action)
    metrics = rollout.compute_metrics_from_action(action)
    assert isinstance(result, RolloutResult)
    assert isinstance(metrics, RolloutMetrics)
    assert result.state.position.shape == action.shape
    assert metrics.actions is action
    result.costs_and_constraints.get_sum_cost().sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert rollout.update_dt(0.01)
    assert rollout.reset(torch.tensor([0]))
    assert rollout.reset_shape()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_rosenbrock_protocol_preserves_mps_device_and_autograd_without_fallback():
    device_cfg = DeviceCfg(torch.device("mps"))
    rollout = RosenbrockRollout(RosenbrockCfg(device_cfg, dimensions=2, time_action_horizon=2))
    action = torch.zeros((1, 2, 2), device="mps", requires_grad=True)
    metrics = rollout.compute_metrics_from_action(action)
    assert metrics.actions.device.type == "mps"
    cost = metrics.costs_and_constraints.get_sum_cost().sum()
    cost.backward()
    assert action.grad is not None and action.grad.device.type == "mps"
