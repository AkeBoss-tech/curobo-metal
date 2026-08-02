"""High-level robot state transition facade."""

from __future__ import annotations

import torch
from typing import Optional, Union
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration, StateFromBSplineKnot, StateFromPositionClique,
    StateFromPositionTeleport, StateFromVelocity,
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
        self.robot_dynamics = self._create_robot_dynamics()
        self.robot_model = self._create_robot_model()
        if self.robot_dynamics is not None:
            self.robot_dynamics.setup_batch_size(config.batch_size, config.horizon)
        self._filter = JointStateFilter(config.state_filter_cfg) if config.state_filter_cfg else None

    def _create_dynamics(self):
        cs = self.config.control_space
        if self.config.teleport_mode:
            return StateFromPositionTeleport(self.config.device_cfg, self.config.batch_size, self.config.horizon)
        if cs == ControlSpace.ACCELERATION:
            return StateFromAcceleration(self.config.device_cfg, self._dt, self._dof,
                                         self.config.batch_size, self.config.horizon)
        if cs == ControlSpace.VELOCITY:
            return StateFromVelocity(self.config.device_cfg, self._dt, self._dof,
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

    def _create_robot_model(self):
        """Compile tree FK once for augmented transition outputs.

        The CUDA implementation owns a packed robot-model buffer.  The
        portable replacement retains the public state lifecycle by composing
        the production whole-body Kinematics implementation instead.
        """
        try:
            from curobo._src.robot.kinematics.kinematics import Kinematics
            from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
            from curobo._src.robot.types import KinematicsParams

            source = self.config.robot_config.kinematics
            params = source if isinstance(source, KinematicsParams) else KinematicsParams(source)
            model_cfg = KinematicsCfg(
                self.config.device_cfg, list(params.tool_frames), params
            )
            return Kinematics(model_cfg, compute_jacobian=False, compute_spheres=True)
        except (AttributeError, TypeError, ValueError):
            # Some low-level callers construct a transition around a cspace
            # only.  They retain tensor-step behavior but cannot request FK.
            return None

    def _create_robot_dynamics(self):
        value = getattr(self.config.robot_config, "dynamics", None)
        if value is None:
            return None
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.robot.types import KinematicsParams

        if isinstance(value, Dynamics):
            return value
        if isinstance(value, DynamicsCfg):
            return Dynamics(value)
        # ``curobo_metal`` robot configs use a tree model as a marker when
        # dynamics are requested.  Recreate the portable public config from
        # the transition robot metadata rather than accepting a CUDA object.
        kin = self.config.robot_config.kinematics
        params = kin if isinstance(kin, KinematicsParams) else KinematicsParams(kin)
        return Dynamics(DynamicsCfg(params, self.config.device_cfg))

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
        state_seq = self.tensor_step(
            start_state, act_seq, None, start_state_idx, goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        return self.compute_augmented_state(state_seq, idxs_env=idxs_env)

    def compute_augmented_state(self, state_seq, idxs_env=None):
        if not isinstance(state_seq, JointState):
            raise TypeError("state_seq must be JointState")
        if state_seq.position.ndim == 1:
            state_seq = state_seq.unsqueeze(0).unsqueeze(1)
        elif state_seq.position.ndim == 2:
            state_seq = state_seq.unsqueeze(1)
        kinematics = None if self.robot_model is None else self.robot_model.compute_kinematics(
            state_seq, idxs_env=idxs_env
        )
        if self.robot_dynamics is None:
            torque = torch.zeros_like(state_seq.position)
        else:
            torque = self.robot_dynamics.compute_inverse_dynamics(state_seq)
        return RobotState(
            joint_state=state_seq, joint_torque=torque,
            cuda_robot_model_state=kinematics,
        )

    def integrate_action(self, act_seq):
        if self.control_space in ControlSpace.position_types():
            return act_seq
        dt = self._dt.to(act_seq)
        if dt.numel() < act_seq.shape[-2]:
            dt = torch.cat((dt, dt[-1:].expand(act_seq.shape[-2] - dt.numel())))
        return torch.cumsum(act_seq * dt[:act_seq.shape[-2]].view(
            *([1] * (act_seq.ndim - 2)), -1, 1), dim=-2)

    def integrate_action_step(self, act, dt):
        return act * dt

    def filter_robot_state(self, current_state):
        return current_state if self._filter is None else self._filter.filter_joint_state(current_state)

    def get_robot_command(self, current_state, act_seq, shift_steps=1, **kwargs):
        del kwargs
        if shift_steps < 1:
            raise ValueError("shift_steps must be positive")
        if self.return_full_act_buffer:
            return self.get_state_from_action(current_state, act_seq)
        if act_seq.shape[-2] < shift_steps:
            raise ValueError("shift_steps exceeds action horizon")
        if self._filter is not None:
            command = current_state
            for step in range(shift_steps):
                command = self._filter.integrate_action(act_seq[..., step, :], command)
            return command
        state = self.forward(current_state, act_seq[..., :shift_steps, :])
        return state.joint_state.get_trajectory_at_horizon_index(shift_steps - 1)

    def get_state_from_action(self, start_state, act_seq, state_idx=None):
        return self.forward(start_state, act_seq, state_idx).joint_state

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
    def init_action_mean(self): return self.get_init_action_mean()
    def get_init_action_mean(self):
        value = self.default_joint_position
        if self.control_space in (ControlSpace.ACCELERATION, ControlSpace.VELOCITY):
            value = torch.zeros_like(value)
        return value.unsqueeze(0).expand(self.action_horizon, -1).clone()
    @property
    def default_joint_position(self):
        return self.device_cfg.to_device(self.config.robot_config.cspace.default_joint_position)
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
