"""High-level robot state transition facade."""

from __future__ import annotations

import torch
from typing import Optional, Union
from curobo._src.state.state_joint import JointState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration, StateFromBSplineKnot, StateFromPositionClique,
    StateFromPositionTeleport,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.util.state_filter import JointStateFilter


class RobotStateTransition:
    def __init__(self, config):
        self.config = config
        self._dt = config.device_cfg.to_device(
            config.dt_traj_params.get_dt_array(max(config.horizon - 1, 1))
        )
        self._dof = len(config.robot_config.cspace.joint_names)
        self._dynamics = self._create_dynamics()
        self.robot_dynamics = None
        dynamics_cfg = getattr(config.robot_config, "dynamics", None)
        if dynamics_cfg is not None:
            from curobo._src.robot.dynamics.dynamics import Dynamics
            self.robot_dynamics = Dynamics(dynamics_cfg)
            self.robot_dynamics.setup_batch_size(config.batch_size, config.horizon)
        self._filter = JointStateFilter(config.state_filter_cfg) if config.state_filter_cfg else None

    def _create_dynamics(self):
        cs = self.config.control_space
        if self.config.teleport_mode:
            return StateFromPositionTeleport(self.config.device_cfg, self.config.batch_size, self.config.horizon)
        if cs == ControlSpace.ACCELERATION:
            return StateFromAcceleration(self.config.device_cfg, self._dt, self._dof,
                                         self.config.batch_size, self.config.horizon)
        if cs in ControlSpace.bspline_types():
            return StateFromBSplineKnot(
                self.config.device_cfg, self._dof, self.config.batch_size,
                self.config.horizon, self.config.n_knots,
                self.config.interpolation_steps, control_space=cs,
            )
        return StateFromPositionClique(self.config.device_cfg, self._dt, self._dof,
                                       batch_size=self.config.batch_size,
                                       horizon=self.config.horizon)

    def _initialize_robot_cmd_state(self):
        self._robot_cmd_state = None

    def update_traj_dt(self, dt, base_dt=None, max_dt=None, base_ratio=None):
        if isinstance(dt, torch.Tensor):
            self._dt = dt.to(**self.config.device_cfg.as_torch_dict())
        else:
            self.config.dt_traj_params.update_dt(dt, base_dt, max_dt, base_ratio)
            self._dt = self.config.device_cfg.to_device(
                self.config.dt_traj_params.get_dt_array(max(self.horizon - 1, 1))
            )
        if hasattr(self._dynamics, "dt_h"):
            self._dynamics.dt_h = self._dt

    def tensor_step(self, state, act, state_seq=None, state_idx=None, **kwargs):
        return self._dynamics.forward(state, act, state_seq, state_idx, **kwargs)

    def robot_cmd_tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState = None,
        state_idx: Optional[torch.Tensor] = None,
        implicit_goal_state: Optional[JointState] = None,
        implicit_goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        return self.tensor_step(
            state, act, state_seq, state_idx,
            goal_state=implicit_goal_state,
            goal_state_idx=implicit_goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

    def update_cmd_batch_size(self, batch_size):
        self.update_batch_size(batch_size)

    def update_batch_size(self, batch_size, force_update=False):
        self.config.batch_size = batch_size
        self._dynamics.update_batch_size(batch_size, force_update=force_update)
        if self.robot_dynamics is not None:
            self.robot_dynamics.setup_batch_size(batch_size, self.horizon)

    def forward(self, start_state, act_seq, start_state_idx=None, goal_state=None,
                goal_state_idx=None, use_implicit_goal_state=None, idxs_env=None):
        del idxs_env
        return self.tensor_step(
            start_state, act_seq, None, start_state_idx, goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

    def compute_augmented_state(self, state_seq, idxs_env=None):
        del idxs_env
        return state_seq

    def integrate_action(self, act_seq):
        return torch.cumsum(act_seq * self._dt[:act_seq.shape[-2]].view(
            *([1] * (act_seq.ndim - 2)), -1, 1), dim=-2)

    def integrate_action_step(self, act, dt):
        return act * dt

    def filter_robot_state(self, current_state):
        return current_state if self._filter is None else self._filter.filter_joint_state(current_state)

    def get_robot_command(self, current_state, act_seq, shift_steps=1, **kwargs):
        del shift_steps, kwargs
        return self.forward(current_state, act_seq).get_trajectory_at_horizon_index(0)

    def get_state_from_action(self, start_state, act_seq, state_idx=None):
        return self.forward(start_state, act_seq, state_idx)

    def get_action_from_state(self, state):
        if self.control_space == ControlSpace.ACCELERATION:
            return state.acceleration
        if self.control_space == ControlSpace.VELOCITY:
            return state.velocity
        return state.position

    @property
    def action_bound_lows(self):
        return -self.action_bound_highs

    @property
    def action_bound_highs(self):
        values = {
            ControlSpace.ACCELERATION: self.max_acceleration,
            ControlSpace.VELOCITY: self.max_velocity,
        }
        return values.get(self.control_space, torch.full((self._dof,), torch.inf, **self.device_cfg.as_torch_dict()))

    @property
    def init_action_mean(self): return self.default_joint_position
    def get_init_action_mean(self): return self.init_action_mean
    @property
    def default_joint_position(self): return self.config.robot_config.cspace.default_joint_position
    @property
    def cspace_distance_weight(self): return self.config.robot_config.cspace.cspace_distance_weight
    @property
    def null_space_weight(self): return self.config.robot_config.cspace.null_space_weight
    @property
    def null_space_maximum_distance(self): return None

    def _limit(self, name, default):
        value = getattr(self.config.robot_config.cspace, name, None)
        if value is None: value = default
        return self.device_cfg.to_device(value if isinstance(value, list) else [value] * self._dof)

    @property
    def max_acceleration(self): return self._limit("max_acceleration", float("inf"))
    @property
    def max_jerk(self): return self._limit("max_jerk", float("inf"))
    @property
    def max_velocity(self): return self._limit("max_velocity", float("inf")) * self.config.vel_scale
    @property
    def action_horizon(self): return self.config.n_knots or self.config.horizon
    @property
    def horizon(self): return self.config.horizon
    @property
    def n_knots(self): return self.config.n_knots
    @property
    def control_space(self): return self.config.control_space
    @property
    def device_cfg(self): return self.config.device_cfg
    @property
    def teleport_mode(self): return self.config.teleport_mode
    @property
    def return_full_act_buffer(self): return self.config.return_full_act_buffer
    @property
    def state_finite_difference_mode(self): return self.config.state_finite_difference_mode
    @property
    def filter_robot_command(self): return self.config.filter_robot_command

    @property
    def compute_inverse_dynamics(self):
        """Whether a dynamics model is configured for this transition.

        The pinned API exposes this as a capability flag; callers that need
        torques should use the configured dynamics facade directly.
        """
        return self.robot_dynamics is not None

    def get_state_bounds(self):
        return self.action_bound_lows, self.action_bound_highs

    def update_link_mass(self, link_name: str, mass: float):
        if self.robot_dynamics is None:
            raise RuntimeError("Cannot update link mass without inverse dynamics")
        self.robot_dynamics.update_link_mass(link_name, mass)

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ):
        if self.robot_dynamics is None:
            raise RuntimeError("Cannot update link inertial properties without inverse dynamics")
        self.robot_dynamics.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]]
    ):
        if self.robot_dynamics is None:
            raise RuntimeError("Cannot update link inertial properties without inverse dynamics")
        self.robot_dynamics.update_links_inertial(link_properties)

    def get_full_dof_from_solution(self, q_js):
        return q_js
