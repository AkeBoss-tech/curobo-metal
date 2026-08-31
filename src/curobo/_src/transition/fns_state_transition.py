"""Differentiable tensor state-transition functions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.curobolib.cuda_ops.trajectory import (
    AccelerationTensorStepIdxKernel,
    BSplineIdxKernel,
    CliqueTensorStepIdxKernel,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


def filter_signal_jit(signal, kernel):
    if signal.ndim != 3:
        raise ValueError("signal must have [batch, horizon, dof] dimensions")
    batch, horizon, dof = signal.shape
    return (
        torch.nn.functional.conv1d(
            signal.transpose(-1, -2).reshape(batch * dof, 1, horizon),
            kernel,
            padding="same",
        )
        .view(batch, dof, horizon)
        .transpose(-1, -2)
        .contiguous()
    )


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
    def __init__(
        self, device_cfg: DeviceCfg, batch_size: int = 1, horizon: int = 1
    ) -> None:
        self.batch_size = -1
        self.horizon = -1
        self.device_cfg = device_cfg
        self._diag_dt = None
        self._inv_dt_h = None
        self.action_horizon = horizon
        self.dt = 0.01
        self.update_batch_size(batch_size, horizon)

    def update_dt(self, dt: float):
        if hasattr(self, "_dt_h") and self._dt_h is not None:
            self._dt_h[:] = dt
        if self._inv_dt_h is not None:
            self._inv_dt_h[:] = 1.0 / dt
        self.dt = dt

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        del force_update
        if batch_size is not None:
            self.batch_size = batch_size
        if horizon is not None:
            self.horizon = horizon

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        raise NotImplementedError


class StateFromPositionTeleport(StateFromBase):
    def __init__(
        self, device_cfg: DeviceCfg, batch_size: int = 1, horizon: int = 1
    ) -> None:
        super().__init__(device_cfg, batch_size=batch_size, horizon=horizon)

    def _forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState = None,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        del start_state_idx, kwargs
        zero = torch.zeros_like(u_act)
        result = JointState(
            u_act, zero, zero, start_state.joint_names, zero,
            dt=torch.as_tensor(self.dt, device=u_act.device),
        )
        return result if out_state_seq is None else out_state_seq.copy_reference(result)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        return self._forward(start_state, u_act, out_state_seq, start_state_idx, **kwargs)


class StateFromAcceleration(StateFromBase):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dt_h: torch.Tensor,
        dof: int,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        self.dof = dof
        self._dt_h = device_cfg.to_device(dt_h)
        self._u_grad = None
        super().__init__(device_cfg, batch_size, horizon)

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        next_batch = self.batch_size if batch_size is None else batch_size
        next_horizon = self.horizon if horizon is None else horizon
        if (
            self._u_grad is None
            or next_batch != self.batch_size
            or next_horizon != self.horizon
        ):
            self._u_grad = torch.zeros(
                (next_batch, next_horizon, self.dof),
                **self.device_cfg.as_torch_dict(),
            )
        if force_update:
            self._u_grad = self._u_grad.detach()
        super().update_batch_size(next_batch, next_horizon)

    def _forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState = None,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        del kwargs
        if start_state_idx is None and out_state_seq is not None:
            raise ValueError("Start state index is required for Acceleration kernel")
        if u_act.ndim < 2 or u_act.shape[-1] != self.dof:
            raise ValueError(f"acceleration action must end in {self.dof} DOF values")
        state = _select_start_state(start_state, u_act.shape[0], start_state_idx)
        if state.velocity is None:
            raise ValueError("acceleration transition requires start-state velocity")
        dt = _step_dt(self._dt_h, u_act.shape[-2], u_act)
        velocity = state.velocity[..., None, :] + torch.cumsum(u_act * dt, dim=-2)
        position = state.position[..., None, :] + torch.cumsum(velocity * dt, dim=-2)
        initial_acceleration = (
            state.acceleration if state.acceleration is not None else torch.zeros_like(state.velocity)
        )
        previous = torch.cat((initial_acceleration[..., None, :], u_act[..., :-1, :]), dim=-2)
        jerk = (u_act - previous) / dt
        result = JointState(position, velocity, u_act, state.joint_names, jerk, dt=self._dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        return self._forward(start_state, u_act, out_state_seq, start_state_idx, **kwargs)


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
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dt_h: torch.Tensor,
        dof: int,
        filter_velocity: bool = False,
        filter_acceleration: bool = False,
        filter_jerk: bool = False,
        batch_size: int = 1,
        horizon: int = 1,
    ) -> None:
        self.dof = dof
        self._dt_h = device_cfg.to_device(dt_h)
        self._inv_dt_h = 1.0 / self._dt_h
        self._u_grad = None
        self._filter_velocity = filter_velocity
        self._filter_acceleration = filter_acceleration
        self._filter_jerk = filter_jerk
        self.filters = filter_velocity, filter_acceleration, filter_jerk
        super().__init__(device_cfg, batch_size, horizon)
        if filter_velocity or filter_acceleration or filter_jerk:
            self._sma_kernel = device_cfg.to_device(
                [[[0.06136, 0.24477, 0.38774, 0.24477, 0.06136]]]
            )

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        next_batch = self.batch_size if batch_size is None else batch_size
        next_horizon = self.horizon if horizon is None else horizon
        next_action_horizon = next_horizon - 4
        if next_action_horizon < 0:
            raise ValueError("Clique horizon must be at least 4")
        if (
            self._u_grad is None
            or next_batch != self.batch_size
            or next_horizon != self.horizon
        ):
            self.action_horizon = next_action_horizon
            self._u_grad = torch.zeros(
                (next_batch, self.action_horizon, self.dof),
                **self.device_cfg.as_torch_dict(),
            )
        if force_update:
            self._u_grad = self._u_grad.detach()
        super().update_batch_size(next_batch, next_horizon)

    def _forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState = None,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        del kwargs
        if start_state_idx is None and out_state_seq is not None:
            raise ValueError("Start state index is required for Clique kernel")
        if goal_state is not None and out_state_seq is not None:
            if goal_state.dt is None or goal_state.dt.shape != goal_state.position.shape[0:2]:
                raise ValueError("Shape mismatch between goal_state.dt and goal_state.position")
            if use_implicit_goal_state is None or use_implicit_goal_state.shape != goal_state.position.shape[0:2]:
                raise ValueError("Shape mismatch for use_implicit_goal_state")
        if u_act.ndim < 2 or u_act.shape[-1] != self.dof:
            raise ValueError(f"position action must end in {self.dof} DOF values")
        state = _select_start_state(start_state, u_act.shape[0], start_state_idx)
        # A CUDA clique action has four guard points.  Accept both that compact
        # representation and an already-expanded [batch, horizon, dof] action
        # sequence so portable planner integrations remain shape-stable.
        output_horizon = getattr(self, "padded_horizon", self.horizon)
        if u_act.shape[-2] < output_horizon:
            missing = output_horizon - u_act.shape[-2]
            head = state.position[..., None, :]
            tail_count = max(0, missing - 1)
            tail = u_act[..., -1:, :].expand(*u_act.shape[:-2], tail_count, self.dof)
            position = torch.cat((head, u_act, tail), dim=-2)
        else:
            position = u_act[..., :output_horizon, :]
        use_goal = goal_state is not None and (
            use_implicit_goal_state is None
            or bool(use_implicit_goal_state.to(dtype=torch.bool).any().item())
        )
        if use_goal:
            goal = _select_start_state(goal_state, position.shape[0], goal_state_idx)
            goal_position = (
                goal.position[..., -1, :] if goal.position.ndim > 2 else goal.position
            )
            position = torch.cat(
                (position[..., :-1, :], goal_position[..., None, :]), dim=-2
            )
        dt = _step_dt(self._dt_h, max(position.shape[-2] - 1, 1), position)[..., :position.shape[-2] - 1, :]
        initial_velocity = state.velocity if state.velocity is not None else torch.zeros_like(state.position)
        velocity = torch.cat(
            (initial_velocity[..., None, :], torch.diff(position, dim=-2) / dt), -2
        )
        acceleration = torch.cat((
            (state.acceleration if state.acceleration is not None else torch.zeros_like(initial_velocity))[..., None, :],
            torch.diff(velocity, dim=-2) / dt,
        ), -2)
        jerk = torch.cat((
            (state.jerk if state.jerk is not None else torch.zeros_like(initial_velocity))[..., None, :],
            torch.diff(acceleration, dim=-2) / dt,
        ), -2)
        if self._filter_velocity:
            velocity = self.filter_signal(velocity)
        if self._filter_acceleration:
            acceleration = self.filter_signal(acceleration)
        if self._filter_jerk:
            jerk = self.filter_signal(jerk)
        result = JointState(position, velocity, acceleration, state.joint_names, jerk, dt=self._dt_h)
        return result if out_state_seq is None else out_state_seq.copy_reference(result)

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        return self._forward(
            start_state,
            u_act,
            out_state_seq,
            start_state_idx,
            goal_state,
            goal_state_idx,
            use_implicit_goal_state,
            **kwargs,
        )

    def filter_signal(self, signal: torch.Tensor):
        return filter_signal_jit(signal, self._sma_kernel)


class StateFromBSplineKnot(StateFromPositionClique):
    def __init__(
        self,
        device_cfg: DeviceCfg,
        dof: int,
        batch_size: int = 1,
        horizon: int = 1,
        n_knots: int = 4,
        interpolation_steps: int = 1,
        use_implicit_goal_state: bool = False,
        control_space: ControlSpace = ControlSpace.BSPLINE_4,
    ) -> None:
        self.dof = dof
        self._u_grad = None
        self.n_knots, self.interpolation_steps = n_knots, interpolation_steps
        self.use_implicit_goal_state = use_implicit_goal_state
        self.control_space = control_space
        self.bspline_degree = ControlSpace.spline_degree(control_space)
        self.padded_horizon = ControlSpace.spline_total_interpolation_steps(
            control_space, n_knots, interpolation_steps
        )
        dt = torch.ones(max(horizon - 1, 1), **device_cfg.as_torch_dict()) * 0.01
        super().__init__(device_cfg, dt, dof, batch_size=batch_size, horizon=horizon)
        self.action_horizon = n_knots
        if self._u_grad is None or self._u_grad.shape != (batch_size, n_knots, dof):
            self._u_grad = torch.zeros(
                (batch_size, n_knots, dof), **device_cfg.as_torch_dict()
            )

    def update_batch_size(
        self,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
        force_update: bool = False,
    ) -> None:
        next_batch = self.batch_size if batch_size is None else batch_size
        next_horizon = self.horizon if horizon is None else horizon
        if (
            self._u_grad is None
            or self._u_grad.shape != (next_batch, self.n_knots, self.dof)
        ):
            self._u_grad = torch.zeros(
                (next_batch, self.n_knots, self.dof),
                **self.device_cfg.as_torch_dict(),
            )
        if force_update:
            self._u_grad = self._u_grad.detach()
        StateFromBase.update_batch_size(self, next_batch, next_horizon)
        self.action_horizon = self.n_knots

    def _forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState = None,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        if out_state_seq is not None:
            if self.use_implicit_goal_state:
                if goal_state is None:
                    raise ValueError("Goal state is not provided for implicit goal state")
                if start_state_idx is not None and goal_state_idx is None:
                    raise ValueError("Goal state index is not provided for implicit goal state")
            else:
                if goal_state is None:
                    goal_state = start_state
                if goal_state_idx is None:
                    goal_state_idx = start_state_idx
            if start_state_idx is None:
                raise ValueError("Start state index is required for BSpline kernel")
            if goal_state_idx is None:
                raise ValueError("Goal state index is None")
            if goal_state.dt is None:
                raise ValueError("goal_state dt is None")
            if use_implicit_goal_state is None:
                raise ValueError("use_implicit_goal_state is None")
            if goal_state_idx.shape[0] != u_act.shape[0]:
                raise ValueError("Shape mismatch between goal_state_idx and action batch")
            if use_implicit_goal_state.shape[0] != goal_state.shape[0]:
                raise ValueError("Shape mismatch for use_implicit_goal_state")
            if start_state.jerk is None:
                raise ValueError("start jerk is None")
            if out_state_seq.dt is None:
                raise ValueError("out_state_seq dt is None")
            if u_act.shape[1] != self.n_knots:
                raise ValueError(
                    f"u_act.shape[1] != self.n_knots: {u_act.shape[1]} != {self.n_knots}"
                )
            if self.padded_horizon != out_state_seq.shape[1]:
                raise ValueError(
                    "padded_horizon does not match out_state_seq horizon: "
                    f"{self.padded_horizon} != {out_state_seq.shape[1]}"
                )
        target_horizon = self.padded_horizon
        if u_act.shape[-2] != target_horizon:
            data = u_act.transpose(-1, -2)
            data = torch.nn.functional.interpolate(
                data, size=target_horizon, mode="linear", align_corners=True
            ).transpose(-1, -2)
        else:
            data = u_act
        return super()._forward(
            start_state,
            data,
            out_state_seq,
            start_state_idx,
            goal_state,
            goal_state_idx,
            use_implicit_goal_state,
            **kwargs,
        )

    def forward(
        self,
        start_state: JointState,
        u_act: torch.Tensor,
        out_state_seq: JointState,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JointState:
        return self._forward(
            start_state,
            u_act,
            out_state_seq,
            start_state_idx,
            goal_state,
            goal_state_idx,
            use_implicit_goal_state,
            **kwargs,
        )


# The backend supports omitted output buffers, while V2's declared CUDA API
# requires them. Keep both: strict declarations above and the portable
# allocating calls at runtime.
StateFromPositionTeleport.forward = StateFromPositionTeleport._forward
StateFromAcceleration.forward = StateFromAcceleration._forward
StateFromPositionClique.forward = StateFromPositionClique._forward
StateFromBSplineKnot.forward = StateFromBSplineKnot._forward
