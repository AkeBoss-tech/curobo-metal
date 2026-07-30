"""Differentiable tensor state-transition functions."""

from __future__ import annotations

import torch
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


def filter_signal_jit(signal, kernel):
    return torch.einsum("...hd,kh->...kd", signal, kernel)


class StateFromBase:
    def __init__(self, device_cfg: DeviceCfg, batch_size: int = 1, horizon: int = 1):
        self.device_cfg, self.batch_size, self.horizon = device_cfg, batch_size, horizon
        self.dt = 0.01

    def update_dt(self, dt: float):
        self.dt = dt

    def update_batch_size(self, batch_size=None, horizon=None, force_update=False):
        del force_update
        if batch_size is not None: self.batch_size = batch_size
        if horizon is not None: self.horizon = horizon

    def forward(self, start_state, u_act, out_state_seq, start_state_idx=None, **kwargs):
        raise NotImplementedError


class StateFromPositionTeleport(StateFromBase):
    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None, **kwargs):
        del start_state_idx, kwargs
        zero = torch.zeros_like(u_act)
        result = JointState(u_act, zero, zero, start_state.joint_names, zero, dt=torch.as_tensor(self.dt))
        return result if out_state_seq is None else out_state_seq.copy_reference(result)


class StateFromAcceleration(StateFromBase):
    def __init__(self, device_cfg, dt_h, dof, batch_size=1, horizon=1):
        super().__init__(device_cfg, batch_size, horizon)
        self.dt_h, self.dof = device_cfg.to_device(dt_h), dof

    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None, **kwargs):
        del start_state_idx, kwargs
        dt = self.dt_h[:u_act.shape[-2]].to(device=u_act.device, dtype=u_act.dtype)
        dt = dt.view(*([1] * (u_act.ndim - 2)), dt.shape[0], 1)
        velocity = start_state.velocity[..., None, :] + torch.cumsum(u_act * dt, dim=-2)
        position = start_state.position[..., None, :] + torch.cumsum(velocity * dt, dim=-2)
        jerk = torch.cat((u_act[..., :1, :] * 0, torch.diff(u_act, dim=-2)), dim=-2) / dt
        result = JointState(position, velocity, u_act, start_state.joint_names, jerk, dt=self.dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)


class StateFromPositionClique(StateFromBase):
    def __init__(self, device_cfg, dt_h, dof, filter_velocity=False,
                 filter_acceleration=False, filter_jerk=False, batch_size=1, horizon=1):
        super().__init__(device_cfg, batch_size, horizon)
        self.dt_h, self.dof = device_cfg.to_device(dt_h), dof
        self.filters = filter_velocity, filter_acceleration, filter_jerk

    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None,
                goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs):
        del start_state_idx, goal_state_idx, use_implicit_goal_state, kwargs
        position = u_act
        if goal_state is not None:
            position = position.clone(); position[..., -1, :] = goal_state.position
        dt = self.dt_h[:position.shape[-2]-1].to(position)
        velocity = torch.cat((start_state.velocity[..., None, :],
                              torch.diff(position, dim=-2) / dt.view(*([1]*(position.ndim-2)), -1, 1)), -2)
        acceleration = torch.cat((velocity[..., :1, :] * 0, torch.diff(velocity, dim=-2)), -2)
        jerk = torch.cat((acceleration[..., :1, :] * 0, torch.diff(acceleration, dim=-2)), -2)
        result = JointState(position, velocity, acceleration, start_state.joint_names, jerk, dt=self.dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)

    def filter_signal(self, signal):
        return signal


class StateFromBSplineKnot(StateFromPositionClique):
    def __init__(self, device_cfg, dof, batch_size=1, horizon=1, n_knots=4,
                 interpolation_steps=1, use_implicit_goal_state=False,
                 control_space=ControlSpace.BSPLINE_4):
        self.n_knots, self.interpolation_steps = n_knots, interpolation_steps
        self.use_implicit_goal_state = use_implicit_goal_state
        self.control_space = control_space
        dt = torch.ones(max(horizon - 1, 1), **device_cfg.as_torch_dict()) * 0.01
        super().__init__(device_cfg, dt, dof, batch_size=batch_size, horizon=horizon)

    def forward(self, start_state, u_act, out_state_seq=None, **kwargs):
        if u_act.shape[-2] != self.horizon:
            data = u_act.transpose(-1, -2)
            data = torch.nn.functional.interpolate(
                data, size=self.horizon, mode="linear", align_corners=True
            ).transpose(-1, -2)
        else:
            data = u_act
        return super().forward(start_state, data, out_state_seq, **kwargs)
