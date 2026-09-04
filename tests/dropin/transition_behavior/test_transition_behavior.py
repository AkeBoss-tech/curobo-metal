"""Portable transition lifecycle coverage beyond CUDA packed-kernel shims."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.transition.fns_state_transition import (
    StateFromPositionClique,
    StateFromVelocity,
)
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.state_filter import FilterCfg


def _transition(control_space=ControlSpace.ACCELERATION, *, dynamics=False):
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml")
    params = kin.kinematics_config
    robot = RobotCfg(params.robot_cfg, DynamicsCfg(params, DeviceCfg()) if dynamics else None)
    return RobotStateTransition(RobotStateTransitionCfg(
        robot_config=robot,
        dt_traj_params=TimeTrajCfg(0.05, 1.0, 0.05),
        device_cfg=DeviceCfg(),
        batch_size=2,
        horizon=5,
        control_space=control_space,
    ))


def test_velocity_state_table_indices_and_autograd():
    transition = StateFromVelocity(DeviceCfg(), torch.full((4,), 0.1), 2, horizon=4)
    state = JointState.from_position(torch.tensor([[0.0, 0.0], [10.0, -2.0]]))
    action = torch.ones(3, 4, 2, requires_grad=True)
    result = transition.forward(state, action, start_state_idx=torch.tensor([1, 0, 1]))
    torch.testing.assert_close(result.position[:, 0], torch.tensor([[10.1, -1.9], [0.1, 0.1], [10.1, -1.9]]))
    result.position.square().sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    with pytest.raises(IndexError, match="out-of-range"):
        transition.forward(state, action.detach(), start_state_idx=torch.tensor([0, 1, 3]))


def test_position_clique_accepts_compact_actions_and_uses_goal():
    state = JointState.from_position(torch.zeros(2, 2))
    model = StateFromPositionClique(DeviceCfg(), torch.full((7,), 0.1), 2, horizon=8)
    compact = torch.ones(2, 4, 2, requires_grad=True)
    goal = JointState.from_position(torch.full((1, 2), 3.0))
    output = model.forward(state, compact, goal_state=goal)
    assert output.position.shape == (2, 8, 2)
    torch.testing.assert_close(output.position[:, 0], torch.zeros(2, 2))
    torch.testing.assert_close(output.position[:, -1], torch.full((2, 2), 3.0))
    output.jerk.square().sum().backward()
    assert compact.grad is not None


def test_transition_augments_kinematics_and_dynamics_and_commands():
    transition = _transition(dynamics=True)
    state = JointState.from_position(
        transition.default_joint_position.repeat(2, 1),
        joint_names=transition.robot_model.joint_names,
    )
    action = torch.zeros(2, 5, 7, requires_grad=True)
    result = transition.forward(state, action)
    assert result.joint_state.position.shape == (2, 5, 7)
    assert result.joint_torque.shape == (2, 5, 7)
    assert result.tool_poses.position.shape[:3] == (2, 5, 1)
    command = transition.get_robot_command(state, action, shift_steps=2)
    assert command.position.shape == (2, 7)
    (result.tool_poses.position.square().sum() + result.joint_torque.square().sum()).backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_velocity_facade_bounds_and_action_mean():
    with pytest.raises(ValueError, match="[Vv]elocity.*not implemented"):
        _transition(ControlSpace.VELOCITY)


def test_transition_uses_joint_limit_bounds_and_keeps_command_lifecycle_separate():
    transition = _transition(ControlSpace.POSITION)
    bounds = transition.get_state_bounds()
    torch.testing.assert_close(transition.action_bound_lows, bounds.position[0])
    torch.testing.assert_close(transition.action_bound_highs, bounds.position[1])

    state = JointState.from_position(
        transition.default_joint_position.unsqueeze(0), joint_names=transition.joint_names
    )
    action = state.position[:, None, :].expand(3, 5, -1).clone()
    output = transition.get_state_from_action(state, action)
    assert output.position.shape == (3, 5, 7)
    assert output.joint_names == transition.joint_names
    assert transition._cmd_batch_size == 3
    # Command expansion must not resize or otherwise alter the rollout batch.
    assert transition.batch_size == 2


def test_transition_updates_schedule_and_validates_action_layout():
    with pytest.raises(ValueError, match="[Vv]elocity.*not implemented"):
        _transition(ControlSpace.VELOCITY)


def test_filtered_multi_step_command_returns_each_command_state():
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml")
    params = kin.kinematics_config
    robot = RobotCfg(params.robot_cfg)
    with pytest.raises(ValueError, match="[Vv]elocity.*not implemented"):
        RobotStateTransition(RobotStateTransitionCfg(
            robot_config=robot,
            dt_traj_params=TimeTrajCfg(0.1, 1.0, 0.1),
            device_cfg=DeviceCfg(),
            batch_size=1,
            horizon=4,
            control_space=ControlSpace.VELOCITY,
            state_filter_cfg=FilterCfg(FilterCoeff(), 0.1, ControlSpace.VELOCITY),
        ))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_transition_mps_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"))
    transition = StateFromVelocity(device, torch.full((3,), 0.1, device="mps"), 2, horizon=3)
    action = torch.ones(1, 3, 2, device="mps", requires_grad=True)
    result = transition.forward(JointState.from_position(torch.zeros(1, 2, device="mps")), action)
    result.position.sum().backward()
    assert result.position.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_transition_facade_mps_command_and_rollout(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"))
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device)
    robot = RobotCfg(kin.kinematics_config)
    with pytest.raises(ValueError, match="[Vv]elocity.*not implemented"):
        RobotStateTransition(RobotStateTransitionCfg(
            robot_config=robot,
            dt_traj_params=TimeTrajCfg(0.05, 1.0, 0.05),
            device_cfg=device,
            batch_size=1,
            horizon=4,
            control_space=ControlSpace.VELOCITY,
        ))
