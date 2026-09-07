"""Portable joint-state filtering and command integration.

This is the stateful, tensor-facing part of cuRobo's command filter.  It is
implemented with ordinary PyTorch operations so a filter can remain on CPU or
Apple MPS without requiring CUDA graph capture or packed CUDA buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import blend_joint_states
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_DOF


@dataclass(frozen=True)
class FilterCfg:
    """Configuration for a persistent joint-command filter.

    ``device_cfg`` intentionally determines the storage device for the
    enabled filter.  A disabled filter is a strict pass-through and leaves its
    input object and device untouched, matching the pinned public behavior.
    """

    filter_coeff: FilterCoeff
    dt: float
    control_space: ControlSpace
    device_cfg: DeviceCfg = DeviceCfg()
    enable: bool = True
    teleport_mode: bool = False

    @staticmethod
    def create(
        coeff_dict,
        enable=True,
        dt=0.0,
        control_space=ControlSpace.ACCELERATION,
        device_cfg=DeviceCfg(),
        teleport_mode=False,
    ):
        """Create a config from the conventional coefficient mapping.

        Accepting a materialized :class:`FilterCoeff` as well is useful for
        configuration loaders that have already parsed the value object.
        """
        coeff = coeff_dict if isinstance(coeff_dict, FilterCoeff) else FilterCoeff(**coeff_dict)
        return FilterCfg(coeff, dt, control_space, device_cfg, enable, teleport_mode)


class JointStateFilter(FilterCfg):
    """Stateful low-pass filtering plus position/velocity/acceleration steps.

    The object owns one command state.  Filtering ingress is converted to the
    configured portable device; explicit states supplied to the integrators
    retain their own device, matching upstream command integration semantics.
    Output tensors retain normal PyTorch autograd edges. Integrators accept a ``[dof]`` action, or any action
    broadcastable to the current state shape, which covers batched and
    batch-by-horizon command buffers without special CUDA kernels.
    """

    def __init__(self, filter_config: FilterCfg):
        if not isinstance(filter_config, FilterCfg):
            raise TypeError("filter_config must be a FilterCfg")
        super().__init__(**vars(filter_config))
        self.cmd_joint_state: Optional[JointState] = None
        if self.control_space == ControlSpace.ACCELERATION:
            self.integrate_action = self.integrate_acc
        elif self.control_space == ControlSpace.VELOCITY:
            self.integrate_action = self.integrate_vel
        elif self.control_space in ControlSpace.position_types():
            self.integrate_action = self.integrate_pos
        else:  # Defensive for deserialized/foreign enum-like values.
            raise ValueError(f"unsupported control space: {self.control_space!r}")

    def filter_joint_state(self, raw_joint_state: JointState):
        """Filter a raw state into the persistent command state.

        Assignment is functional rather than an in-place ``copy_`` so a
        caller can differentiate through a sequence of filtering operations.
        The ``JointState`` instance itself remains stable after its first
        ingress, preserving code that holds command-state buffer references.
        """
        if not isinstance(raw_joint_state, JointState):
            raise TypeError("raw_joint_state must be a JointState")
        if not self.enable:
            return raw_joint_state

        raw_joint_state = raw_joint_state.to(self.device_cfg)
        if self.cmd_joint_state is None:
            self.cmd_joint_state = raw_joint_state.clone()
            return self.cmd_joint_state

        self._require_compatible_state(raw_joint_state)
        for name in ("position", "velocity", "acceleration", "jerk"):
            old, new = getattr(self.cmd_joint_state, name), getattr(raw_joint_state, name)
            if old is None or new is None:
                continue
            weight = getattr(self.filter_coeff, name)
            setattr(self.cmd_joint_state, name, weight * new + (1.0 - weight) * old)
        return self.cmd_joint_state

    def _require_compatible_state(self, state: JointState) -> None:
        if self.cmd_joint_state is None:
            return
        current = self.cmd_joint_state.position
        incoming = state.position
        if current.shape != incoming.shape:
            raise ValueError(
                "state shape changed from "
                f"{tuple(current.shape)} to {tuple(incoming.shape)}; call reset() before reconfiguration"
            )
        if current.device != incoming.device or current.dtype != incoming.dtype:
            raise ValueError("state device or dtype changed; call reset() before reconfiguration")

    def _set_state(self, state: Optional[JointState]) -> JointState:
        """Select an explicit command state or the persistent one.

        Explicit states are copied so subsequent integration never aliases a
        caller-owned input.  Reusing an existing compatible buffer keeps the
        same object identity, consistent with solver command-buffer lifecycle.
        """
        if state is not None:
            if not isinstance(state, JointState):
                raise TypeError("cmd_joint_state must be a JointState")
            if self.cmd_joint_state is None:
                self.cmd_joint_state = state.clone()
            else:
                self._require_compatible_state(state)
                self.cmd_joint_state.copy_(state)
        if self.cmd_joint_state is None:
            raise ValueError("cmd_joint_state is required before integrating an action")
        return self.cmd_joint_state

    def _dt(self, dt: Optional[float | torch.Tensor], *, nonzero: bool = False) -> torch.Tensor:
        state = self.cmd_joint_state
        if state is None:  # Kept explicit for static analysis and future callers.
            raise ValueError("cmd_joint_state is required")
        value = self.dt if dt is None else dt
        value = torch.as_tensor(value, device=state.position.device, dtype=state.position.dtype)
        try:
            torch.broadcast_shapes(tuple(state.position.shape[:-1]), tuple(value.shape))
        except RuntimeError as error:
            raise ValueError(
                f"dt shape {tuple(value.shape)} is not broadcastable to command prefix "
                f"{tuple(state.position.shape[:-1])}"
            ) from error
        if nonzero and bool(torch.any(value == 0).detach().cpu()):
            raise ValueError("dt must be nonzero for non-teleport position integration")
        return value.unsqueeze(-1) if value.ndim < state.position.ndim else value

    def _action(self, action: T_DOF, name: str) -> torch.Tensor:
        state = self.cmd_joint_state
        if state is None:
            raise ValueError("cmd_joint_state is required")
        value = torch.as_tensor(action, device=state.position.device, dtype=state.position.dtype)
        if value.ndim == 0:
            return value.expand_as(state.position)
        if value.shape[-1] not in (1, state.position.shape[-1]):
            raise ValueError(
                f"{name} final dimension must be 1 or {state.position.shape[-1]}, got {value.shape[-1]}"
            )
        try:
            return torch.broadcast_to(value, state.position.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not broadcastable to "
                f"command shape {tuple(state.position.shape)}"
            ) from error

    def _derivative(self, name: str) -> torch.Tensor:
        state = self.cmd_joint_state
        if state is None:
            raise ValueError("cmd_joint_state is required")
        value = getattr(state, name)
        if value is None:
            value = torch.zeros_like(state.position)
            setattr(state, name, value)
        return value

    def integrate_jerk(
        self,
        qddd_des,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        state = self._set_state(cmd_joint_state)
        step = self._dt(dt)
        jerk = self._action(qddd_des, "qddd_des")
        state.acceleration = self._derivative("acceleration") + jerk * step
        state.velocity = self._derivative("velocity") + state.acceleration * step
        state.position = state.position + state.velocity * step
        state.jerk = jerk
        return state

    def integrate_acc(
        self,
        qdd_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        state = self._set_state(cmd_joint_state)
        step = self._dt(dt)
        acceleration = self._action(qdd_des, "qdd_des")
        state.acceleration = acceleration
        state.velocity = self._derivative("velocity") + acceleration * step
        state.position = state.position + state.velocity * step
        state.jerk = torch.zeros_like(acceleration)
        # Pinned cuRobo returns a clone for acceleration commands; retain that
        # observable ownership boundary while the internal command state stays
        # persistent for the next action.
        return state.clone()

    def integrate_vel(
        self,
        qd_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        state = self._set_state(cmd_joint_state)
        step = self._dt(dt)
        velocity = self._action(qd_des, "qd_des")
        state.velocity = velocity
        state.position = state.position + velocity * step
        return state

    def integrate_pos(
        self,
        q_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        state = self._set_state(cmd_joint_state)
        position = self._action(q_des, "q_des")
        if not self.teleport_mode:
            step = self._dt(dt, nonzero=True)
            state.velocity = (position - state.position) / step
        state.position = position
        return state

    def reset(self):
        """Discard persistent command buffers before a shape/device change."""
        self.cmd_joint_state = None
