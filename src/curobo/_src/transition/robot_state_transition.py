"""High-level robot state transition facade."""

from __future__ import annotations

import torch
import torch.autograd.profiler as profiler
from typing import TYPE_CHECKING, Optional, Union
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float16_tensors, check_float32_tensors
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import augment_joint_state, stack_joint_states
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration, StateFromBSplineKnot, StateFromPositionClique,
    StateFromPositionTeleport, StateFromVelocity,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.util.state_filter import JointStateFilter
from curobo._src.util.logging import log_and_raise, log_info

if TYPE_CHECKING:
    from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg


class _RobotStateTransitionPortableMixin:
    @property
    def action_order(self):
        if self.control_space in ControlSpace.position_types():
            return 0
        return 1 if self.control_space == ControlSpace.VELOCITY else 2


class RobotStateTransition(_RobotStateTransitionPortableMixin):
    def __init__(self, config: RobotStateTransitionCfg):
        self.config = config
        if not hasattr(config, "robot_config") or not hasattr(config, "device_cfg"):
            raise TypeError("config must be a RobotStateTransitionCfg-compatible record")
        if config.control_space == ControlSpace.VELOCITY:
            log_and_raise("Velocity control space not implemented for RobotStateTransition")
        self.batch_size = config.batch_size
        self.interpolation_steps = config.interpolation_steps
        self._dt = config.device_cfg.to_device(
            config.dt_traj_params.get_dt_array(max(config.horizon, 1))
        )
        # Keep the public V2 schedule/value fields as ordinary tensors.  They
        # intentionally are not CUDA graph buffers: every transition keeps
        # its own reusable metadata while differentiable forwards construct a
        # fresh result rather than retaining an old optimizer graph.
        self.traj_dt = self._dt
        self.dt = float(config.dt_traj_params.base_dt)
        self._dof = len(config.robot_config.cspace.joint_names)
        self._dynamics = self._create_dynamics()
        # Upstream owns distinct CUDA buffers for planner rollouts and robot
        # commands.  Keeping two lightweight tensor transition instances is
        # important even without the packed CUDA ABI: command generation must
        # not mutate a differentiable optimizer rollout's lifecycle state.
        self._cmd_dynamics = self._create_dynamics()
        self._rollout_step_fn = self._dynamics
        self._cmd_step_fn = self._cmd_dynamics
        self.robot_dynamics = self._create_robot_dynamics()
        self.robot_model = self._create_robot_model()
        self.num_dof = self._dof
        self.d_dof = self._dof
        self.action_dim = self._dof
        self.d_state = 4 * self._dof
        self.joint_names = (
            list(self.robot_model.joint_names)
            if self.robot_model is not None
            else list(config.robot_config.cspace.joint_names)
        )
        if self.robot_dynamics is not None:
            self.robot_dynamics.setup_batch_size(config.batch_size, config.horizon)
        self._filter = JointStateFilter(config.state_filter_cfg) if config.state_filter_cfg else None
        self.joint_limits = self.get_state_bounds()
        self.state_seq = JointState.zeros(
            (self.batch_size, self.horizon, self.action_dim), self.device_cfg, self.joint_names
        )
        self.Z = self.device_cfg.to_device([0.0])
        self._initialize_robot_cmd_state()

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
            dynamics = StateFromBSplineKnot(
                self.config.device_cfg, self._dof, self.config.batch_size,
                self.config.horizon, self.config.n_knots,
                self.config.interpolation_steps, control_space=cs,
            )
            dynamics._dt_h = self._dt
            dynamics._inv_dt_h = 1.0 / self._dt
            return dynamics
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
            if isinstance(source, KinematicsCfg):
                return Kinematics(
                    source, compute_jacobian=False, compute_spheres=True
                )
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
        # This is a reusable *shape* buffer only.  Results are copied by
        # reference by the differentiable transition functions, so no graph is
        # retained between optimizer iterations.
        self._robot_cmd_state_seq = JointState.zeros(
            (1, self.horizon, self.action_dim), self.device_cfg, self.joint_names
        )
        self._cmd_batch_size = 1

    def update_traj_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        if isinstance(dt, torch.Tensor):
            schedule = dt.to(**self.config.device_cfg.as_torch_dict()).reshape(-1)
            if schedule.numel() == 0 or bool((schedule <= 0).any().item()):
                raise ValueError("trajectory timestep schedule must be non-empty and positive")
            if not bool(torch.isfinite(schedule).all().item()):
                raise ValueError("trajectory timestep schedule must be finite")
            self._dt = schedule
            # A scalar tensor has the same meaning as V2's uniform ``all_dt``
            # form.  For a non-uniform caller-owned schedule, retain exact
            # tensor values and only expose its leading timestep as ``dt``.
            if schedule.numel() == 1:
                self.config.dt_traj_params.update_dt(all_dt=float(schedule.detach().cpu()))
            self.dt = float(schedule[0].detach().cpu())
        else:
            self.config.dt_traj_params.update_dt(dt, base_dt, max_dt, base_ratio)
            self._dt = self.config.device_cfg.to_device(
                self.config.dt_traj_params.get_dt_array(max(self.horizon, 1))
            )
            self.dt = float(self.config.dt_traj_params.base_dt)
        self.traj_dt = self._dt
        for dynamics in (self._dynamics, self._cmd_dynamics):
            if hasattr(dynamics, "dt_h"):
                dynamics.dt_h = self._dt
            elif hasattr(dynamics, "update_dt"):
                dynamics.update_dt(self._dt)

    def tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        self._validate_step_inputs(state, act)
        return self._dynamics.forward(
            state,
            act,
            state_seq,
            state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

    def robot_cmd_tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
        implicit_goal_state: Optional[JointState] = None,
        implicit_goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        self._validate_step_inputs(state, act)
        result = self._cmd_dynamics.forward(
            state, act, state_seq, state_idx,
            goal_state=implicit_goal_state,
            goal_state_idx=implicit_goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        result.joint_names = self.joint_names.copy()
        return result

    def _validate_step_inputs(self, state: JointState, act: torch.Tensor) -> None:
        """Reject accidental host/device or joint-layout mixing early.

        CUDA's packed kernels reject these inputs implicitly.  Explicit
        validation gives CPU/MPS callers a useful error instead of a later
        device copy or obscure matrix failure.
        """
        if not isinstance(state, JointState):
            raise TypeError("state must be JointState")
        if not isinstance(act, torch.Tensor):
            raise TypeError("act must be a torch.Tensor")
        if act.ndim < 2 or act.shape[-1] != self.action_dim:
            raise ValueError(f"act must end in {self.action_dim} DOF values")
        if state.position.shape[-1] != self.action_dim:
            raise ValueError(f"state must end in {self.action_dim} DOF values")
        if act.device != state.position.device:
            raise ValueError("act and state must be on the same device")
        if act.dtype != state.position.dtype:
            raise ValueError("act and state must use the same dtype")
        if not self.device_cfg.is_same_torch_device(act.device):
            raise ValueError("state and act must reside on config.device_cfg.device")

    def update_cmd_batch_size(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size != self._cmd_batch_size:
            self._robot_cmd_state_seq = JointState.zeros(
                (batch_size, self.horizon, self.action_dim), self.device_cfg, self.joint_names
            )
            self._cmd_dynamics.update_batch_size(batch_size, self.horizon)
            self._cmd_batch_size = batch_size

    def update_batch_size(self, batch_size, force_update=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config.batch_size = batch_size
        self.batch_size = batch_size
        self._dynamics.update_batch_size(batch_size, force_update=force_update)
        if self.state_seq.shape[0] != batch_size or force_update:
            self.state_seq = JointState.zeros(
                (batch_size, self.horizon, self.action_dim), self.device_cfg, self.joint_names
            )
        if self.robot_dynamics is not None:
            self.robot_dynamics.setup_batch_size(batch_size, self.horizon)

    def forward(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        idxs_env: Optional[torch.Tensor] = None,
    ) -> RobotState:
        if not isinstance(act_seq, torch.Tensor) or act_seq.ndim != 3:
            raise ValueError("act_seq must have [batch, horizon, dof] dimensions")
        self._validate_step_inputs(start_state, act_seq)
        self.update_batch_size(act_seq.shape[0], force_update=act_seq.requires_grad)
        state_seq = self.tensor_step(
            start_state, act_seq, None, start_state_idx, goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        return self.compute_augmented_state(state_seq, idxs_env=idxs_env)

    def compute_augmented_state(
        self, state_seq: JointState, idxs_env: Optional[torch.Tensor] = None
    ) -> RobotState:
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
        if not isinstance(act_seq, torch.Tensor) or act_seq.ndim < 2:
            raise ValueError("act_seq must include horizon and DOF dimensions")
        if act_seq.shape[-1] != self.action_dim:
            raise ValueError(f"act_seq must end in {self.action_dim} DOF values")
        if self.control_space in ControlSpace.position_types():
            return act_seq
        dt = self._dt.to(act_seq)
        if dt.numel() < act_seq.shape[-2]:
            dt = torch.cat((dt, dt[-1:].expand(act_seq.shape[-2] - dt.numel())))
        step = dt[:act_seq.shape[-2]].view(*([1] * (act_seq.ndim - 2)), -1, 1)
        velocity = torch.cumsum(act_seq * step, dim=-2)
        if self.control_space == ControlSpace.VELOCITY:
            return velocity
        # Acceleration actions integrate once into velocity and again into
        # position.  This replaces the CUDA integration matrix with normal
        # differentiable PyTorch operations on CPU/MPS.
        return torch.cumsum(velocity * step, dim=-2)

    def integrate_action_step(self, act, dt):
        if not isinstance(act, torch.Tensor):
            raise TypeError("act must be a torch.Tensor")
        if self.control_space in ControlSpace.position_types():
            return act
        value = act * dt
        return value if self.control_space == ControlSpace.VELOCITY else value * dt

    def filter_robot_state(self, current_state: JointState):
        return current_state if self._filter is None else self._filter.filter_joint_state(current_state)

    def get_robot_command(
        self,
        current_state: JointState,
        act_seq: torch.Tensor,
        shift_steps: int = 1,
        state_idx: Optional[torch.Tensor] = None,
        implicit_goal_state: Optional[JointState] = None,
        implicit_goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        if shift_steps < 1:
            raise ValueError("shift_steps must be positive")
        if self.return_full_act_buffer:
            self.update_cmd_batch_size(act_seq.shape[0])
            return self.robot_cmd_tensor_step(
                current_state, act_seq, self._robot_cmd_state_seq,
                state_idx, implicit_goal_state,
                implicit_goal_state_idx, use_implicit_goal_state,
            )
        if act_seq.shape[-2] < shift_steps:
            raise ValueError("shift_steps exceeds action horizon")
        if self._filter is not None:
            command = current_state
            command_buffer = None
            for step in range(shift_steps):
                command = self._filter.integrate_action(act_seq[..., step, :], command)
                command_buffer = (
                    command.clone()
                    if command_buffer is None
                    else stack_joint_states(command_buffer, command)
                )
            return command if shift_steps == 1 else command_buffer
        state = self.forward(current_state, act_seq[..., :shift_steps, :])
        return state.joint_state.get_trajectory_at_horizon_index(shift_steps - 1)

    def get_state_from_action(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        self.update_cmd_batch_size(act_seq.shape[0])
        if state_idx is None:
            state_idx = torch.zeros(
                act_seq.shape[0], device=act_seq.device, dtype=torch.int32
            )
        return self.robot_cmd_tensor_step(
            start_state, act_seq, self._robot_cmd_state_seq, state_idx
        )

    def get_action_from_state(self, state: JointState) -> torch.Tensor:
        if self.control_space == ControlSpace.ACCELERATION:
            return state.acceleration
        if self.control_space == ControlSpace.VELOCITY:
            return state.velocity
        return state.position

    @property
    def action_bound_lows(self):
        bounds = self.get_state_bounds()
        if self.control_space in ControlSpace.position_types():
            return bounds.position[0]
        if self.control_space == ControlSpace.VELOCITY:
            return bounds.velocity[0]
        return bounds.acceleration[0]

    @property
    def action_bound_highs(self):
        bounds = self.get_state_bounds()
        if self.control_space in ControlSpace.position_types():
            return bounds.position[1]
        if self.control_space == ControlSpace.VELOCITY:
            return bounds.velocity[1]
        return bounds.acceleration[1]

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
    def null_space_maximum_distance(self):
        return getattr(self.config.robot_config.cspace, "null_space_maximum_distance", None)
    def _limit(self, name, default):
        value = getattr(self.config.robot_config.cspace, name, None)
        if value is None:
            value = default
        if isinstance(value, torch.Tensor):
            result = self.device_cfg.to_device(value).reshape(-1)
            if result.numel() == 1:
                result = result.expand(self._dof)
            if result.numel() != self._dof:
                raise ValueError(f"{name} must contain one value or {self._dof} DOF values")
            return result
        if not isinstance(value, (list, tuple)):
            value = [value] * self._dof
        return self.device_cfg.to_device(value)

    @property
    def max_acceleration(self): return self.get_state_bounds().acceleration[1]
    @property
    def max_jerk(self): return self.get_state_bounds().jerk[1]
    @property
    def max_velocity(self): return self.get_state_bounds().velocity[1] * self.config.vel_scale
    @property
    def action_horizon(self): return self._rollout_step_fn.action_horizon
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
        if self.robot_model is not None:
            bounds = self.robot_model.get_joint_limits().clone()
        else:
            # A cspace-only transition has no URDF limits.  It still exposes a
            # well-formed JointLimits record for optimizer/component callers.
            from curobo._src.robot.types import JointLimits
            finite = self.device_cfg.to_device(self._limit("max_acceleration", 10.0))
            jerk = self.device_cfg.to_device(self._limit("max_jerk", 500.0))
            velocity = torch.full_like(finite, float("inf"))
            position = torch.full_like(finite, float("inf"))
            bounds = JointLimits(
                self.joint_names.copy(), torch.stack((-position, position)),
                torch.stack((-velocity, velocity)), torch.stack((-finite, finite)),
                torch.stack((-jerk, jerk)), None, self.device_cfg,
            )
        cspace = self.config.robot_config.cspace
        return cspace.scale_joint_limits(bounds) if hasattr(cspace, "scale_joint_limits") else bounds

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

    def get_full_dof_from_solution(self, q_js: JointState) -> JointState:
        if not isinstance(q_js, JointState):
            raise TypeError("q_js must be JointState")
        return q_js if self.robot_model is None else self.robot_model.get_full_js(q_js)
