from __future__ import annotations

import pytest
import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost as CanonicalToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCost, ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def _poses(device="cpu"):
    current = ToolPose(["left", "right"], torch.zeros(2, 2, 2, 3, device=device), torch.tensor([1.0, 0, 0, 0], device=device).repeat(2, 2, 2, 1))
    goal = GoalToolPose(["left", "right"], torch.zeros(2, 2, 2, 1, 3, device=device), torch.tensor([1.0, 0, 0, 0], device=device).repeat(2, 2, 2, 1, 1))
    return current, goal


def test_config_preserves_explicit_criteria_and_clone_is_independent():
    criterion = ToolPoseCriteria.track_position([1.0, 0.0, 0.0])
    cfg = ToolPoseCostCfg(weight=[1.0, 2.0], tool_frames=["left", "right"], tool_pose_criteria={"left": criterion})
    assert cfg.class_type is CanonicalToolPoseCost
    torch.testing.assert_close(cfg.tool_pose_criteria["left"].terminal_pose_axes_weight_factor, criterion.terminal_pose_axes_weight_factor)
    clone = cfg.clone()
    clone.tool_pose_criteria["left"].terminal_pose_axes_weight_factor.zero_()
    assert bool(cfg.tool_pose_criteria["left"].terminal_pose_axes_weight_factor.any())
    cfg.set_tool_frames(["right", "left", "camera"])
    assert cfg.tool_frames == ["right", "left", "camera"]
    assert cfg.num_links == 3 and cfg.rotation_method == 0


def test_config_rejects_invalid_frame_and_criterion_ownership():
    with pytest.raises(ValueError, match="unique"):
        ToolPoseCostCfg(weight=1.0, tool_frames=["tool", "tool"])
    with pytest.raises(ValueError, match="unknown"):
        ToolPoseCostCfg(weight=1.0, tool_frames=["tool"], tool_pose_criteria={"other": ToolPoseCriteria.disabled()})
    with pytest.raises(TypeError, match="ToolPoseCriteria"):
        ToolPoseCostCfg(weight=1.0, tool_frames=["tool"], tool_pose_criteria={"tool": object()})


def test_cost_validates_goal_batch_indices_and_keeps_autograd_composable():
    cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["left", "right"])
    current, goal = _poses()
    current.position.requires_grad_()
    cost, _, _, indices = ToolPoseCost(cfg)(current, goal, torch.tensor([1, 0]))
    assert cost.shape == (2, 2, 2)
    assert indices.dtype == torch.int32
    cost.sum().backward()
    assert current.position.grad is not None and torch.isfinite(current.position.grad).all()
    with pytest.raises(ValueError, match="out-of-range"):
        ToolPoseCost(cfg)(current, goal, torch.tensor([2, 0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_cost_cfg_and_tensors_stay_on_mps_without_fallback():
    cfg = ToolPoseCostCfg(weight=1.0, tool_frames=["left", "right"])
    current, goal = _poses("mps")
    result, *_ = ToolPoseCost(cfg)(current, goal, torch.tensor([0, 1], device="mps"))
    assert result.device.type == "mps"
