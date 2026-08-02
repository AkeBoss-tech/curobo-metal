"""Differentiable tensor state-transition functions."""

from __future__ import annotations

import torch
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


def filter_signal_jit(signal, kernel):
    return torch.einsum("...hd,kh->...kd", signal, kernel)


def _select_start_state(start_state: JointState, batch_size: int, indices=None) -> JointState:
    """Select and broadcast initial states for a transition batch.

    The CUDA kernels take a packed state table plus an int32 index per
    rollout.  Keeping this operation in ordinary PyTorch gives CPU and MPS
    callers the same useful table/index lifecycle without pretending to use
    the packed CUDA ABI.
    """
    if start_state.position.ndim == 1:
        position = start_state.position.unsqueeze(0).expand(batch_size, -1)
        def value(field):
            tensor = getattr(start_state, field)
            return None if tensor is None else tensor.unsqueeze(0).expand(batch_size, -1)
    else:
        table_size = start_state.position.shape[0]
        if indices is None:
            if table_size == batch_size:
                indices = torch.arange(batch_size, device=start_state.position.device)
            elif table_size == 1:
                indices = torch.zeros(batch_size, device=start_state.position.device, dtype=torch.long)
            else:
                raise ValueError(
                    "start_state has multiple rows; start_state_idx is required for this action batch"
                )
        indices = torch.as_tensor(indices, device=start_state.position.device, dtype=torch.long).reshape(-1)
        if indices.numel() != batch_size:
            raise ValueError("start_state_idx must contain one index per action batch")
        if bool(((indices < 0) | (indices >= table_size)).any().item()):
            raise IndexError("start_state_idx contains an out-of-range state index")
        position = start_state.position.index_select(0, indices)
        def value(field):
            tensor = getattr(start_state, field)
            return None if tensor is None else tensor.index_select(0, indices)
    return JointState(
        position, value("velocity"), value("acceleration"), start_state.joint_names,
        value("jerk"), dt=start_state.dt,
    )


def _step_dt(dt_h: torch.Tensor, count: int, reference: torch.Tensor) -> torch.Tensor:
    """Return ``count`` positive time steps on the action device/dtype."""
    if count < 1:
        return reference.new_empty((0,))
    dt = dt_h.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if dt.numel() == 0:
        raise ValueError("transition timestep schedule cannot be empty")
    if bool((dt <= 0).any().item()):
        raise ValueError("transition timestep schedule must be strictly positive")
    if dt.numel() < count:
        dt = torch.cat((dt, dt[-1:].expand(count - dt.numel())))
    return dt[:count].view(*([1] * (reference.ndim - 2)), count, 1)


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
        result = JointState(
            u_act, zero, zero, start_state.joint_names, zero,
            dt=torch.as_tensor(self.dt, device=u_act.device),
        )
        return result if out_state_seq is None else out_state_seq.copy_reference(result)


class StateFromAcceleration(StateFromBase):
    def __init__(self, device_cfg, dt_h, dof, batch_size=1, horizon=1):
        super().__init__(device_cfg, batch_size, horizon)
        self.dt_h, self.dof = device_cfg.to_device(dt_h), dof

    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None, **kwargs):
        del kwargs
        if u_act.ndim < 2 or u_act.shape[-1] != self.dof:
            raise ValueError(f"acceleration action must end in {self.dof} DOF values")
        state = _select_start_state(start_state, u_act.shape[0], start_state_idx)
        if state.velocity is None:
            raise ValueError("acceleration transition requires start-state velocity")
        dt = _step_dt(self.dt_h, u_act.shape[-2], u_act)
        velocity = state.velocity[..., None, :] + torch.cumsum(u_act * dt, dim=-2)
        position = state.position[..., None, :] + torch.cumsum(velocity * dt, dim=-2)
        initial_acceleration = (
            state.acceleration if state.acceleration is not None else torch.zeros_like(state.velocity)
        )
        previous = torch.cat((initial_acceleration[..., None, :], u_act[..., :-1, :]), dim=-2)
        jerk = (u_act - previous) / dt
        result = JointState(position, velocity, u_act, state.joint_names, jerk, dt=self.dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)


class StateFromVelocity(StateFromBase):
    """Integrate velocity commands with differentiable CPU/MPS tensor ops."""

    def __init__(self, device_cfg, dt_h, dof, batch_size=1, horizon=1):
        super().__init__(device_cfg, batch_size, horizon)
        self.dt_h, self.dof = device_cfg.to_device(dt_h), dof

    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None, **kwargs):
        del kwargs
        if u_act.ndim < 2 or u_act.shape[-1] != self.dof:
            raise ValueError(f"velocity action must end in {self.dof} DOF values")
        state = _select_start_state(start_state, u_act.shape[0], start_state_idx)
        dt = _step_dt(self.dt_h, u_act.shape[-2], u_act)
        position = state.position[..., None, :] + torch.cumsum(u_act * dt, dim=-2)
        previous = torch.cat((state.velocity[..., None, :], u_act[..., :-1, :]), dim=-2)
        acceleration = (u_act - previous) / dt
        jerk_previous = torch.cat((
            (state.acceleration if state.acceleration is not None else torch.zeros_like(u_act[..., 0, :]))[..., None, :],
            acceleration[..., :-1, :],
        ), dim=-2)
        jerk = (acceleration - jerk_previous) / dt
        result = JointState(position, u_act, acceleration, state.joint_names, jerk, dt=self.dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)


class StateFromPositionClique(StateFromBase):
    def __init__(self, device_cfg, dt_h, dof, filter_velocity=False,
                 filter_acceleration=False, filter_jerk=False, batch_size=1, horizon=1):
        super().__init__(device_cfg, batch_size, horizon)
        self.dt_h, self.dof = device_cfg.to_device(dt_h), dof
        self.filters = filter_velocity, filter_acceleration, filter_jerk

    def forward(self, start_state, u_act, out_state_seq=None, start_state_idx=None,
                goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs):
        del goal_state_idx, use_implicit_goal_state, kwargs
        if u_act.ndim < 2 or u_act.shape[-1] != self.dof:
            raise ValueError(f"position action must end in {self.dof} DOF values")
        state = _select_start_state(start_state, u_act.shape[0], start_state_idx)
        # A CUDA clique action has four guard points.  Accept both that compact
        # representation and an already-expanded [batch, horizon, dof] action
        # sequence so portable planner integrations remain shape-stable.
        if u_act.shape[-2] < self.horizon:
            missing = self.horizon - u_act.shape[-2]
            head = state.position[..., None, :]
            tail_count = max(0, missing - 1)
            tail = u_act[..., -1:, :].expand(*u_act.shape[:-2], tail_count, self.dof)
            position = torch.cat((head, u_act, tail), dim=-2)
        else:
            position = u_act[..., :self.horizon, :]
        if goal_state is not None:
            goal = _select_start_state(goal_state, position.shape[0], None)
            position = torch.cat((position[..., :-1, :], goal.position[..., None, :]), dim=-2)
        dt = _step_dt(self.dt_h, max(position.shape[-2] - 1, 1), position)[..., :position.shape[-2] - 1, :]
        initial_velocity = state.velocity if state.velocity is not None else torch.zeros_like(state.position)
        velocity = torch.cat((initial_velocity[..., None, :],
                              torch.diff(position, dim=-2) / dt.view(*([1]*(position.ndim-2)), -1, 1)), -2)
        acceleration = torch.cat((
            (state.acceleration if state.acceleration is not None else torch.zeros_like(initial_velocity))[..., None, :],
            torch.diff(velocity, dim=-2) / dt,
        ), -2)
        jerk = torch.cat((
            (state.jerk if state.jerk is not None else torch.zeros_like(initial_velocity))[..., None, :],
            torch.diff(acceleration, dim=-2) / dt,
        ), -2)
        result = JointState(position, velocity, acceleration, state.joint_names, jerk, dt=self.dt_h)
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
