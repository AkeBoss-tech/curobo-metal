"""Portable value-lifecycle coverage for pinned rollout metric containers."""

import pytest
import torch

from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _metrics(*, requires_grad: bool = False) -> RolloutMetrics:
    actions = torch.arange(24.0).reshape(2, 3, 4).requires_grad_(requires_grad)
    costs = CostCollection()
    costs.add(actions[..., :1], "objective", weight=torch.ones(2, 1, 1))
    constraints = CostCollection()
    constraints.add(-torch.ones(2, 3, 1), "constraint")
    return RolloutMetrics(
        actions=actions,
        costs_and_constraints=CostsAndConstraints(costs=costs, constraints=constraints),
        state=JointState.from_position(actions),
        feasible=torch.tensor([[True, False, True], [False, True, True]]),
        convergence=CostCollection([actions[..., 1:2]], ["goal"]),
        debug={"nested": {"trace": actions[..., 2:3]}, "label": "portable"},
    )


def test_to_device_cfg_preserves_bool_metadata_and_copies_nested_rollout_channels():
    metrics = _metrics()
    converted = metrics.to(DeviceCfg(device="cpu", dtype=torch.float64))

    assert converted is not metrics
    assert converted.actions.dtype is torch.float64
    assert converted.costs_and_constraints.costs.values[0].dtype is torch.float64
    assert converted.state.position.dtype is torch.float64
    assert converted.convergence.values[0].dtype is torch.float64
    assert converted.feasible.dtype is torch.bool
    assert converted.debug["nested"]["trace"].dtype is torch.float64
    assert metrics.actions.dtype is torch.float32


def test_clone_detach_and_empty_length_are_value_safe():
    metrics = _metrics(requires_grad=True)
    clone = metrics.clone()
    detached = metrics.detach()

    clone.debug["nested"]["trace"].zero_()
    assert bool(metrics.debug["nested"]["trace"].any())
    assert clone.actions is not metrics.actions
    assert detached.actions.requires_grad is False
    assert detached.state.position.requires_grad is False
    assert detached.costs_and_constraints.costs.values[0].requires_grad is False
    assert len(RolloutResult()) == 0


def test_merge_normalizes_legacy_compact_weight_lists_before_appending():
    left = CostCollection([torch.ones(1, 2, 1)], ["plain"])
    right = CostCollection()
    right.add(torch.ones(1, 2, 1), "weighted", weight=torch.full((1, 1, 1), 2.0))
    left.merge(right)

    assert left.names == ["plain", "weighted"]
    assert left.weights[0] is None
    torch.testing.assert_close(left.weights[1], torch.full((1, 1, 1), 2.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_to_mps_keeps_every_tensor_channel_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    metrics = _metrics().to(DeviceCfg(device="mps", dtype=torch.float32))

    assert metrics.actions.device.type == "mps"
    assert metrics.state.position.device.type == "mps"
    assert metrics.costs_and_constraints.get_sum_cost(True).device.type == "mps"
    assert metrics.feasible.device.type == "mps"
    assert metrics.debug["nested"]["trace"].device.type == "mps"
