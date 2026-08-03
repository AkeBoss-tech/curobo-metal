"""Portable lifecycle coverage for ToolPose-family tensor values."""

from __future__ import annotations

import pytest
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def _quaternion(*shape: int, device: str = "cpu") -> torch.Tensor:
    value = torch.randn(*shape, 4, device=device)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def _tool(device: str = "cpu", *, batch: int = 2, horizon: int = 3) -> ToolPose:
    return ToolPose(
        ["base", "tool"],
        torch.randn(batch, horizon, 2, 3, device=device),
        _quaternion(batch, horizon, 2, device=device),
    )


def _goal(device: str = "cpu", *, goalset: int = 2) -> GoalToolPose:
    return GoalToolPose(
        ["base", "tool"],
        torch.randn(2, 3, 2, goalset, 3, device=device),
        _quaternion(2, 3, 2, goalset, device=device),
    )


def test_tool_pose_preserves_rank_for_scalar_and_tensor_batch_indexing() -> None:
    tool = _tool()
    assert tool[1].shape == (1, 3, 2, 3)
    assert tool[torch.tensor(1)].shape == (1, 3, 2, 3)
    assert tool[1:].shape == (1, 3, 2, 3)
    with pytest.raises(IndexError, match="batch dimension"):
        tool[..., 0]


def test_tool_pose_validates_full_tensor_contract() -> None:
    with pytest.raises(ValueError, match=r"\[B,H,L,3\]"):
        ToolPose(["tool"], torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 4))
    with pytest.raises(ValueError, match="leading dimensions"):
        ToolPose(["tool"], torch.zeros(1, 1, 1, 3), torch.zeros(2, 1, 1, 4))
    with pytest.raises(ValueError, match="duplicate"):
        ToolPose(["tool", "tool"], torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 4))
    with pytest.raises(TypeError, match="floating"):
        ToolPose(["tool"], torch.zeros(1, 1, 1, 3, dtype=torch.int64), torch.zeros(1, 1, 1, 4, dtype=torch.int64))

    # Empty link sets are meaningful for a caller-owned output buffer and
    # retain regular batched tensor semantics.
    empty = ToolPose([], torch.zeros(2, 1, 0, 3), torch.zeros(2, 1, 0, 4))
    assert empty.to_dict() == {}


def test_tool_pose_transform_and_autograd_value_lifecycle() -> None:
    tool = _tool().requires_grad_()
    selected = tool.reorder_links(["tool"])
    assert selected.tool_frames == ["tool"]
    goal = selected.as_goal()
    assert goal.shape == (2, 3, 1, 1, 3)
    assert torch.equal(goal.get_goalset(0).position, selected.position)

    extracted = selected.get_link_pose("tool")
    (extracted.position.square().sum() + extracted.quaternion.square().sum()).backward()
    assert tool.position.grad is not None
    assert tool.quaternion.grad is not None

    detached = tool.detach()
    clone = tool.clone()
    assert not detached.position.requires_grad
    clone.position.add_(1.0)
    assert not torch.equal(clone.position, tool.position)
    assert tool.to(DeviceCfg(dtype=torch.float64)).dtype == torch.float64
    with pytest.raises(ValueError, match="either device_cfg"):
        tool.to(DeviceCfg(), dtype=torch.float64)


def test_goal_from_poses_validates_goalset_layout_and_noncontiguous_input() -> None:
    position = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).transpose(0, 1).reshape(4, 3)
    quaternion = _quaternion(4)
    value = GoalToolPose.from_poses({"tool": Pose(position, quaternion, name="tool")}, num_goalset=2)
    assert value.shape == (2, 1, 1, 2, 3)
    assert value.get_goalset(1).shape == (2, 1, 1, 3)
    with pytest.raises(ValueError, match="divisible"):
        GoalToolPose.from_poses({"tool": Pose(position[:3], quaternion[:3], name="tool")}, num_goalset=2)
    with pytest.raises(IndexError, match="goalset_index"):
        value.get_goalset(2)


def test_goal_value_lifecycle_rejects_incompatible_destination_buffers() -> None:
    goal = _goal()
    target = _goal()
    assert target.copy_(goal) is target
    assert torch.equal(target.position, goal.position)
    with pytest.raises(ValueError, match="matching"):
        _goal(goalset=1).copy_(goal)
    assert goal[torch.tensor(0)].shape == (1, 3, 2, 2, 3)
    assert goal.contiguous().shape == goal.shape


def test_sequence_frame_views_slices_and_value_lifecycle() -> None:
    sequence = SequenceGoalToolPose(
        ["tool"],
        torch.randn(4, 2, 1, 2, 3),
        _quaternion(4, 2, 1, 2),
    ).requires_grad_()
    frame = sequence.get_frame(-1)
    assert frame.shape == (2, 1, 1, 2, 3)
    assert torch.equal(frame.position[:, 0], sequence.position[-1])
    assert sequence[1:3].shape == (2, 2, 1, 2, 3)
    (frame.position.square().sum() + frame.quaternion.square().sum()).backward()
    assert sequence.position.grad is not None
    assert sequence.quaternion.grad is not None
    detached = sequence.detach()
    assert not detached.position.requires_grad
    copied = sequence.clone().detach()
    assert copied.copy_(detached) is copied
    with pytest.raises(IndexError, match="frame dimension"):
        sequence[..., 0]
    with pytest.raises(IndexError, match="outside"):
        sequence.get_frame(4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_tool_pose_lifecycle_without_cpu_fallback() -> None:
    tool = _tool("mps").requires_grad_()
    goal = tool.as_goal().reorder_links(["tool"])
    sequence = SequenceGoalToolPose(
        ["tool"], goal.position.transpose(0, 1), goal.quaternion.transpose(0, 1)
    )
    output = sequence.get_frame(1).get_goalset(0).get_link_pose("tool")
    output.position.square().sum().backward()
    assert output.device.type == "mps"
    assert tool.position.grad is not None and tool.position.grad.device.type == "mps"
