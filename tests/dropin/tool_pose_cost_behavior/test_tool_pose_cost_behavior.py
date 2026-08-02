"""Behavioral coverage for the public portable ToolPoseCost implementation."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def _current(*, device="cpu", dtype=torch.float32, horizon=3):
    position = torch.tensor(
        [[[[0.3, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.1, 0.0, 0.0]]]],
        device=device, dtype=dtype,
    )[:, :horizon].requires_grad_()
    quat = torch.zeros((1, horizon, 1, 4), device=device, dtype=dtype)
    quat[..., 0] = 1.0
    return ToolPose(["tool"], position, quat.requires_grad_())


def _goals(*, device="cpu", dtype=torch.float32, batch=1, horizon=1):
    position = torch.zeros((batch, horizon, 1, 2, 3), device=device, dtype=dtype)
    position[..., 1, 0] = 1.0
    quat = torch.zeros((batch, horizon, 1, 2, 4), device=device, dtype=dtype)
    quat[..., 0] = 1.0
    return GoalToolPose(["tool"], position.requires_grad_(), quat.requires_grad_())


def test_interleaved_goalset_weighted_cost_and_broadcast_horizon() -> None:
    current = _current()
    goals = _goals(horizon=1)
    cfg = ToolPoseCostCfg(weight=[2.0, 3.0], tool_frames=["tool"])
    cost = ToolPoseCost(cfg)
    assert cost.setup_batch_tensors(1, 3)
    output, position_distance, rotation_distance, indices = cost(current, goals)
    assert output.shape == (1, 3, 2)
    assert position_distance.shape == rotation_distance.shape == indices.shape == (1, 3, 1)
    # The terminal position is closer to the first goal, and its 0.1
    # displacement uses
    # 0.5 * position_weight * displacement^2.
    torch.testing.assert_close(output[0, -1, 0], torch.tensor(0.01))
    torch.testing.assert_close(output[..., 1], torch.zeros_like(output[..., 1]))
    assert torch.equal(indices, torch.zeros_like(indices))
    assert cost._out_distance.shape == (1, 3, 2)
    torch.testing.assert_close(cost._out_distance, output.detach())
    output.sum().backward()
    assert torch.isfinite(current.position.grad).all()
    assert torch.isfinite(goals.position.grad).all()


def test_batch_goal_indices_criteria_and_lie_group_orientation() -> None:
    current = ToolPose(
        ["tool"],
        torch.zeros((2, 2, 1, 3), requires_grad=True),
        torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]]).expand(2, -1, -1, -1).clone().requires_grad_(),
    )
    goals = GoalToolPose(
        ["tool"],
        torch.tensor([[[[[0.0, 0.0, 0.0]]]], [[[[2.0, 0.0, 0.0]]]]], requires_grad=True),
        torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]], [[[[0.0, 0.0, 0.0, 1.0]]]]], requires_grad=True),
    )
    cfg = ToolPoseCostCfg(weight=[1.0, 1.0], tool_frames=["tool"], use_lie_group=True)
    cfg.tool_pose_criteria["tool"] = ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 0.0, 0.0], rpy=[0.0, 0.0, 1.0], non_terminal_scale=1.0
    )
    cost = ToolPoseCost(cfg)
    output, _, angle, _ = cost(current, goals, torch.tensor([[1], [0]], dtype=torch.int32))
    # batch 0 selects the rotated, translated goal; batch 1 selects identity.
    assert output[0, :, 0].gt(0).all() and output[0, :, 1].gt(0).all()
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    torch.testing.assert_close(angle[0], torch.full_like(angle[0], torch.pi))
    output.sum().backward()
    assert torch.isfinite(current.position.grad).all()
    assert torch.isfinite(current.quaternion.grad).all()


def test_tolerance_updates_and_input_validation() -> None:
    current, goals = _current(horizon=1), _goals(horizon=1)
    cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["tool"])
    cost = ToolPoseCost(cfg)
    cost.update_tool_pose_criteria({"tool": ToolPoseCriteria(terminal_pose_convergence_tolerance=[1.0, 1.0])})
    output, *_ = cost(current, goals)
    assert torch.equal(output, torch.zeros_like(output))
    with pytest.raises(IndexError, match="out-of-range"):
        cost(current, goals, torch.tensor([2], dtype=torch.int32))
    with pytest.raises(ValueError, match="goal horizon"):
        cost(current, _goals(horizon=2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tool_pose_cost_mps_fallback_disabled_and_autograd(monkeypatch) -> None:
    # The test runner exports PYTORCH_ENABLE_MPS_FALLBACK=0; retain a runtime
    # guard so this test is also useful when invoked directly.
    del monkeypatch
    cfg = ToolPoseCostCfg(weight=[1.0, 2.0], tool_frames=["tool"], device_cfg=DeviceCfg(device="mps"))
    current, goals = _current(device="mps"), _goals(device="mps")
    output, _, _, _ = ToolPoseCost(cfg)(current, goals)
    assert output.device.type == "mps"
    output.sum().backward()
    assert current.position.grad is not None and torch.isfinite(current.position.grad).all()
