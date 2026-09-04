"""Portable RobotStateTransition lifecycle and numerical contract tests."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


def _cfg(
    control_space: ControlSpace = ControlSpace.ACCELERATION,
    *,
    device_cfg: DeviceCfg = DeviceCfg(),
    horizon: int = 4,
) -> RobotStateTransitionCfg:
    kinematics = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
    return RobotStateTransitionCfg(
        robot_config=RobotCfg(kinematics.kinematics_config),
        dt_traj_params=TimeTrajCfg(0.1, 1.0, 0.1),
        device_cfg=device_cfg,
        batch_size=2,
        horizon=horizon,
        control_space=control_space,
    )


def test_time_schedule_handles_zero_and_horizon_one_and_validates_updates():
    schedule = TimeTrajCfg(0.1, 1.0, 0.1)
    assert schedule.get_dt_array(0) == []
    assert schedule.get_dt_array(1) == [0.1]
    with pytest.raises(ValueError, match="positive"):
        schedule.update_dt(all_dt=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TimeTrajCfg(0.1, 1.01, 0.1)


def test_create_accepts_materialized_values_and_preserves_optional_filter_absence():
    cfg = _cfg()
    created = RobotStateTransitionCfg.create(
        {
            "dt_traj_params": cfg.dt_traj_params,
            "control_space": "velocity",
            "batch_size": 3,
            "horizon": 2,
        },
        cfg.robot_config,
        cfg.device_cfg,
    )
    assert created.control_space is ControlSpace.VELOCITY
    assert created.state_filter_cfg is None
    assert created.batch_size == 3
    with pytest.raises(ValueError, match="unknown control_space"):
        RobotStateTransitionCfg.create(
            {"dt_traj_params": {"base_dt": 0.1, "base_ratio": 1.0, "max_dt": 0.1}, "control_space": "warp"},
            cfg.robot_config,
        )


def test_transition_exposes_stable_schedule_and_rebuilds_public_batch_buffer():
    transition = RobotStateTransition(_cfg(ControlSpace.ACCELERATION, horizon=1))
    assert transition.traj_dt.shape == (1,)
    assert transition.dt == pytest.approx(0.1)
    assert transition.state_seq.position.shape == (2, 1, 7)
    assert transition.joint_limits.dof == 7
    assert transition.action_order == 2
    transition.update_batch_size(5)
    assert transition.state_seq.position.shape == (5, 1, 7)
    transition.update_traj_dt(torch.tensor([0.25]))
    assert transition.dt == pytest.approx(0.25)
    torch.testing.assert_close(transition.traj_dt, torch.tensor([0.25]))


def test_integrate_action_uses_control_order_and_has_finite_gradients():
    acceleration = RobotStateTransition(_cfg(ControlSpace.ACCELERATION))
    action = torch.ones(1, 3, 7, requires_grad=True)
    acceleration_result = acceleration.integrate_action(action)
    torch.testing.assert_close(acceleration_result[0, :, 0], torch.tensor([0.01, 0.03, 0.06]))
    acceleration_result.square().sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()

    with pytest.raises(ValueError, match="Velocity control space not implemented"):
        RobotStateTransition(_cfg(ControlSpace.VELOCITY))


def test_transition_rejects_wrong_rank_or_config_device():
    transition = RobotStateTransition(_cfg(ControlSpace.ACCELERATION))
    state = JointState.from_position(torch.zeros(1, 7), joint_names=transition.joint_names)
    with pytest.raises(ValueError, match=r"\[batch, horizon, dof\]"):
        transition.forward(state, torch.zeros(4, 7))
    with pytest.raises(ValueError, match="same dtype"):
        transition.forward(state, torch.zeros(1, 4, 7, dtype=torch.float64))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_transition_schedule_and_double_integration_stay_on_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"))
    transition = RobotStateTransition(_cfg(ControlSpace.ACCELERATION, device_cfg=device_cfg))
    action = torch.ones(1, 4, 7, device="mps", requires_grad=True)
    output = transition.integrate_action(action)
    assert output.device.type == "mps"
    output.sum().backward()
    assert action.grad is not None and action.grad.device.type == "mps"
