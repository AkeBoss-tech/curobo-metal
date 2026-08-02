"""Behavioral coverage for the portable V2 C-space distance facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.types.device_cfg import DeviceCfg


def _cost(device: str = "cpu") -> CSpaceDistCost:
    cfg = CSpaceDistCostCfg(
        weight=2.0,
        dof=2,
        only_terminal_cost=False,
        terminal_dof_weight=[3.0, 2.0],
        non_terminal_dof_weight=[1.0, 4.0],
        device_cfg=DeviceCfg(torch.device(device)),
    )
    return CSpaceDistCost(cfg)


def test_allocated_rollout_matches_weighted_indexed_formula_and_records_buffers() -> None:
    cost = _cost()
    assert cost.setup_batch_tensors(2, 3)
    q = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        ],
        requires_grad=True,
    )
    goals = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    idx = torch.tensor([0, 1], dtype=torch.int64)
    actual, distance = cost.forward_out_distance(q, goals, idx)
    selected = goals[idx].unsqueeze(1)
    dof_weight = torch.tensor([[[1.0, 4.0], [1.0, 4.0], [3.0, 2.0]]])
    components = 2.0 * (q - selected).square() * dof_weight
    expected = components.sum(dim=-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(distance.square(), expected / 2.0)
    assert actual.shape == (2, 3)
    assert cost._out_cv_buffer is not None and cost._out_g_buffer is not None
    torch.testing.assert_close(cost._out_cv_buffer, components.detach())
    actual.sum().backward()
    torch.testing.assert_close(q.grad, 4.0 * (q.detach() - selected.detach()) * dof_weight)
    assert goals.grad is not None and torch.isfinite(goals.grad).all()


def test_reset_and_validation_keep_buffer_lifecycle_explicit() -> None:
    cost = _cost()
    cost.setup_batch_tensors(2, 2)
    q = torch.ones((2, 2, 2))
    goal = torch.zeros((2, 2))
    assert cost(q, goal, torch.tensor([0, 1])).shape == (2, 2)
    cost.reset(torch.tensor([1], dtype=torch.int32))
    assert torch.count_nonzero(cost._out_cv_buffer[1]) == 0
    assert torch.count_nonzero(cost._out_g_buffer[1]) == 0
    assert torch.count_nonzero(cost._out_cv_buffer[0]) > 0
    with pytest.raises(ValueError, match="horizon"):
        cost(torch.ones((2, 3, 2)), goal, torch.tensor([0, 1]))
    with pytest.raises(TypeError, match="int32 or int64"):
        cost(q, goal, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="out-of-range"):
        cost(q, goal, torch.tensor([0, 2]))


def test_unallocated_direct_calls_retain_legacy_component_shape() -> None:
    cost = _cost()
    q = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    result = cost(q, torch.zeros((1, 2)), torch.tensor([0]))
    assert result.shape == q.shape
    result.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_allocated_rollout_stays_on_mps_without_cpu_fallback() -> None:
    cost = _cost("mps")
    cost.setup_batch_tensors(1, 2)
    q = torch.tensor([[[0.2, -0.3], [0.5, 0.1]]], device="mps", requires_grad=True)
    result = cost(q, torch.zeros((1, 2), device="mps"), torch.tensor([0], device="mps"))
    result.sum().backward()
    assert result.shape == (1, 2)
    assert result.device.type == q.grad.device.type == "mps"
    assert cost._out_cv_buffer.device.type == cost._out_g_buffer.device.type == "mps"


def test_unspecified_dof_is_resolved_when_an_allocated_rollout_arrives() -> None:
    cost = CSpaceDistCost(CSpaceDistCostCfg(weight=1.0))
    cost.setup_batch_tensors(1, 2)
    value = cost(torch.ones((1, 2, 3)), torch.zeros((1, 3)), torch.tensor([0]))
    assert value.shape == (1, 2)
    assert cost.config.dof == 3
    assert cost._out_cv_buffer.shape == (1, 2, 3)
