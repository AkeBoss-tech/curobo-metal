"""Portable lifecycle coverage for the pinned V2 GoalManager surface."""

import pytest
import torch

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _solve(goalset=3, seeds=2):
    return SolveState(SolveMode.BATCH, 2, 1, num_goalset=goalset, num_ik_seeds=seeds, tool_frames=["tool"])


def _poses(device, goalset=3, offset=0.0):
    position = torch.zeros(2, 1, 1, goalset, 3, device=device)
    position[..., 0] = torch.arange(goalset, device=device) + offset
    quaternion = torch.zeros(2, 1, 1, goalset, 4, device=device)
    quaternion[..., 0] = 1.0
    return GoalToolPose(["tool"], position, quaternion)


def _state(device, value=0.0):
    return JointState.from_position(torch.full((2, 3), value, device=device))


def test_goal_buffer_value_update_reuses_reference_and_copies_payloads():
    manager = GoalManager(DeviceCfg("cpu"))
    solve = _solve()
    first, changed = manager.update_goal_buffer(
        solve, goal_tool_poses=_poses("cpu"), current_js=_state("cpu"), goal_js=_state("cpu", 2.0)
    )
    assert changed
    updated, changed = manager.update_goal_buffer(
        solve, goal_tool_poses=_poses("cpu", offset=10.0), current_js=_state("cpu", 4.0), goal_js=_state("cpu", 5.0)
    )
    assert not changed
    assert updated is first
    assert updated.link_goal_poses.position[0, 0, 0, :, 0].tolist() == [10.0, 11.0, 12.0]
    assert updated.current_js.position[0, 0].item() == 4.0
    assert updated.goal_js.position[0, 0].item() == 5.0
    assert manager.batch_helper[:, 0].tolist() == [0, 1]


def test_smaller_goalset_pads_and_reuses_larger_cache():
    manager = GoalManager(DeviceCfg("cpu"))
    first, changed = manager.update_goal_buffer(_solve(3), goal_tool_poses=_poses("cpu", 3))
    assert changed
    second, changed = manager.update_goal_buffer(_solve(1), goal_tool_poses=_poses("cpu", 1, offset=9.0))
    assert not changed
    assert second is first
    # Pinned V2 selects the first new goal for every unused cached slot.
    assert second.link_goal_poses.position[0, 0, 0, :, 0].tolist() == [9.0, 9.0, 9.0]
    assert manager.solve_state.num_goalset == 3


def test_registry_install_copies_values_without_replacing_reference():
    manager = GoalManager(DeviceCfg("cpu"))
    solve = _solve(goalset=1)
    source = GoalRegistry(
        goal_js=_state("cpu", 1.0), current_js=_state("cpu", 2.0),
        link_goal_poses=_poses("cpu", 1),
    )
    first, changed = manager.update_from_goal_registry(solve, source)
    assert changed
    replacement = GoalRegistry(
        goal_js=_state("cpu", 7.0), current_js=_state("cpu", 8.0),
        link_goal_poses=_poses("cpu", 1, offset=3.0),
    )
    second, changed = manager.update_from_goal_registry(solve, replacement)
    assert not changed
    assert second is first
    assert second.goal_js.position[0, 0].item() == 7.0
    assert second.current_js.position[0, 0].item() == 8.0


def test_shape_frame_and_cross_device_fail_explicitly():
    manager = GoalManager(DeviceCfg("cpu"))
    manager.update_goal_buffer(_solve(1), goal_tool_poses=_poses("cpu", 1), current_js=_state("cpu"))
    with pytest.raises(ValueError, match="goal link poses"):
        pose = _poses("cpu", 1)
        manager.update_goal_tool_poses(GoalToolPose(["other"], pose.position, pose.quaternion))
    with pytest.raises(ValueError, match="current state"):
        manager.update_current_state(JointState.from_position(torch.zeros(3, 3)))
    if torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="move inputs explicitly"):
            manager.update_goal_state(_state("mps"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_goal_manager_mps_update_stays_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    manager = GoalManager(DeviceCfg(torch.device("mps")))
    solve = _solve(goalset=2)
    initial, changed = manager.update_goal_buffer(
        solve, goal_tool_poses=_poses("mps", 2), current_js=_state("mps"),
        seed_goal_js=JointState.from_position(torch.zeros(2, 2, 3, device="mps")),
        use_implicit_goal=True,
    )
    assert changed
    updated, changed = manager.update_goal_buffer(
        solve, goal_tool_poses=_poses("mps", 2, offset=3.0), current_js=_state("mps", 4.0),
        seed_goal_js=JointState.from_position(torch.ones(2, 2, 3, device="mps")),
        use_implicit_goal=True,
    )
    assert not changed and updated is initial
    assert updated.link_goal_poses.position.device.type == "mps"
    assert updated.seed_goal_js.position.device.type == "mps"
    assert bool(updated.seed_enable_implicit_goal_js.all().item())
