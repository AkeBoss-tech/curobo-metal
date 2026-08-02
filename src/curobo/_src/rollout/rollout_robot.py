"""Portable robot rollout over the CPU/MPS transition and cost backends.

The upstream class captures a number of CUDA graphs and evaluates cost terms
on multiple streams.  This implementation deliberately keeps the *rollout*
semantics -- goal expansion, state transition, separated cost/constraint
collections, convergence metrics, deterministic action samples, and manager
lifecycle -- while using ordinary PyTorch execution on the caller's device.
Raw CUDA graph, stream, and packed-buffer interfaces remain unsupported.
"""

from __future__ import annotations

from typing import Optional

import torch

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.robot_state_transition import RobotStateTransition

from .cost_manager.cost_manager_robot import RobotCostManager
from .rollout_robot_cfg import RobotRolloutCfg


class RobotRollout:
    """Forward-simulate candidate actions and score robot trajectory terms.

    ``use_cuda_graph=True`` is accepted so upstream configurations can load,
    but no graph is created.  Normal calls continue to use eager CPU/MPS
    PyTorch; callers that explicitly reset/capture CUDA graphs receive a
    precise unsupported-feature error.
    """

    def __init__(self, config: Optional[RobotRolloutCfg] = None,
                 scene_collision_checker=None, use_cuda_graph: bool = False):
        self.config = config
        self.scene_collision_checker = scene_collision_checker
        self._use_cuda_graph = bool(use_cuda_graph)
        self._num_particles_goal: Optional[GoalRegistry] = None
        self._metrics_goal: Optional[GoalRegistry] = None
        # The CUDA implementation keeps graph-address-stable goal buffers.
        # Eager execution has no address-stability constraint, but retaining
        # a layout signature lets us reuse those buffers for value-only goal
        # updates and rebuild them precisely when a solve changes shape.
        self._particle_goal_layout = None
        self._particle_multiplier: Optional[int] = None
        self.start_state: Optional[JointState] = None
        self._batch_size: Optional[int] = None
        self.rollout_instance_name: Optional[str] = None
        self._sample_generator = torch.Generator(device="cpu")

        self.transition_model = None
        self.metrics_transition_model = None
        self.cost_manager = None
        self.constraint_manager = None
        self.hybrid_cost_constraint_manager = None
        self.metrics_cost_manager = None
        self.metrics_constraint_manager = None
        self.metrics_hybrid_cost_constraint_manager = None
        self.metrics_convergence_manager = None
        self._cost_manager_list = []

        if config is None:
            self.device_cfg = None
            self.sum_horizon = False
            self.sampler_seed = 1312
        else:
            self.device_cfg = config.device_cfg
            self.sum_horizon = bool(config.sum_horizon)
            self.sampler_seed = int(config.sampler_seed)
            self._initialize_components()
        self.reset_seed()

    def _new_transition(self):
        cfg = self.config.transition_model_cfg
        if cfg is None:
            return None
        cls = getattr(cfg, "class_type", None) or RobotStateTransition
        return cls(cfg)

    @staticmethod
    def _new_manager(cfg, device_cfg):
        cls = getattr(cfg, "class_type", None) or RobotCostManager
        return cls(device_cfg)

    def _initialize_components(self):
        self.transition_model = self._new_transition()
        self.metrics_transition_model = self._new_transition()
        for cfg_name, manager_name, metrics_name in (
            ("cost_cfg", "cost_manager", "metrics_cost_manager"),
            ("constraint_cfg", "constraint_manager", "metrics_constraint_manager"),
            ("hybrid_cost_constraint_cfg", "hybrid_cost_constraint_manager",
             "metrics_hybrid_cost_constraint_manager"),
        ):
            cfg = getattr(self.config, cfg_name)
            if cfg is None:
                continue
            manager = self._new_manager(cfg, self.device_cfg)
            metrics = self._new_manager(cfg, self.device_cfg)
            manager.initialize_from_config(cfg, self.transition_model, self.scene_collision_checker)
            metrics.initialize_from_config(cfg, self.metrics_transition_model, self.scene_collision_checker)
            setattr(self, manager_name, manager)
            setattr(self, metrics_name, metrics)
            self._cost_manager_list.extend((manager, metrics))
        cfg = self.config.convergence_cfg
        if cfg is not None:
            manager = self._new_manager(cfg, self.device_cfg)
            manager.initialize_from_config(cfg, self.metrics_transition_model, self.scene_collision_checker)
            self.metrics_convergence_manager = manager
            self._cost_manager_list.append(manager)

    @property
    def action_dim(self):
        if self.transition_model is not None:
            return self.transition_model.action_dim
        return 0

    @property
    def action_horizon(self):
        if self.transition_model is not None:
            return self.transition_model.action_horizon
        return 1

    @property
    def horizon(self):
        if self.transition_model is not None:
            return self.transition_model.horizon
        return self.action_horizon

    @property
    def dt(self):
        if self.transition_model is None:
            return 1.0
        value = getattr(self.transition_model, "_dt", None)
        return float(value.reshape(-1)[0].item()) if isinstance(value, torch.Tensor) else 1.0

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.update_batch_size(value)

    @property
    def action_bound_lows(self):
        return None if self.transition_model is None else self.transition_model.action_bound_lows

    @property
    def action_bound_highs(self):
        return None if self.transition_model is None else self.transition_model.action_bound_highs

    @property
    def action_bounds(self):
        if self.transition_model is None:
            return (None, None)
        return torch.stack((self.action_bound_lows, self.action_bound_highs))

    @property
    def state_bounds(self):
        return self.action_bounds

    @property
    def default_joint_position(self):
        return None if self.transition_model is None else self.transition_model.default_joint_position

    @property
    def default_joint_state(self):
        return self.default_joint_position

    # These intentionally stay callable for compatibility with the previous
    # Metal package release.  They always report false because CUDA graphs are
    # never silently emulated on a different execution backend.
    def valid_compute_metrics_from_state_cuda_graph(self):
        return False

    def valid_compute_metrics_from_action_cuda_graph(self):
        return False

    def _fallback_start_state(self, act_seq):
        return JointState.from_position(torch.zeros_like(act_seq[..., 0, :]))

    def _validate_action(self, act_seq):
        """Validate an action batch without inserting an implicit device copy.

        A CPU action passed to an MPS rollout otherwise fails deep inside a
        cost term (and an implicit ``to`` would sever the caller's autograd
        and residency expectations).  The config-less convenience rollout is
        deliberately permissive about its action dimension/horizon because it
        is used as a simple tensor-shaped rollout in public smoke tests.
        """
        if not isinstance(act_seq, torch.Tensor) or act_seq.ndim != 3:
            raise ValueError("act_seq must have shape [batch, horizon, action_dim]")
        if not act_seq.is_floating_point():
            raise TypeError("act_seq must have a floating-point dtype")
        if self.device_cfg is not None and not self.device_cfg.is_same_torch_device(act_seq.device):
            raise ValueError(
                "action device does not match rollout device: "
                f"{act_seq.device} != {self.device_cfg.device}"
            )
        if self.transition_model is not None:
            if act_seq.shape[-1] != self.action_dim:
                raise ValueError(
                    f"action_dim mismatch: expected {self.action_dim}, got {act_seq.shape[-1]}"
                )
            if act_seq.shape[-2] != self.action_horizon:
                raise ValueError(
                    f"action horizon mismatch: expected {self.action_horizon}, got {act_seq.shape[-2]}"
                )
        return act_seq

    @staticmethod
    def _goal_batch_size(goal: GoalRegistry) -> int:
        """Get a concrete problem-batch size for partly constructed goals."""
        # A pose/joint target owns the planning-problem dimension.  In the
        # current-state-only form used by low-level callers, however, the
        # dataclass field is derived only once; prefer the live state so a
        # caller can replace it with a differently sized batch between
        # solves.
        if goal.link_goal_poses is not None or goal.goal_js is not None:
            if goal.batch_size >= 0:
                return int(goal.batch_size)
        for state in (goal.current_js, goal.seed_goal_js):
            if state is not None:
                return int(state.position.shape[0])
        if goal.batch_size >= 0:
            return int(goal.batch_size)
        raise ValueError("goal has no state or pose payload from which to infer batch size")

    @staticmethod
    def _goal_layout(goal: GoalRegistry):
        """Metadata that changes the persistent particle-buffer shape."""
        def shape(name):
            value = getattr(goal, name, None)
            return None if value is None else tuple(value.shape)

        return (
            int(goal.batch_size), int(goal.num_seeds),
            shape("idxs_link_pose"), shape("idxs_goal_js"),
            shape("idxs_current_js"), shape("idxs_seed_goal_js"),
            shape("idxs_enable"), shape("idxs_env"),
        )

    def _compute_state_from_action_impl(self, act_seq):
        self._validate_action(act_seq)
        if self.transition_model is None:
            return JointState.from_position(act_seq)
        goal = self._num_particles_goal
        return self.transition_model.forward(
            self.start_state or self._fallback_start_state(act_seq), act_seq,
            None if goal is None else goal.idxs_current_js,
            None if goal is None else goal.seed_goal_js,
            None if goal is None else goal.idxs_seed_goal_js,
            None if goal is None else goal.seed_enable_implicit_goal_js,
            idxs_env=None if goal is None else goal.idxs_env,
        )

    def _compute_state_from_action_metrics_impl(self, act_seq):
        self._validate_action(act_seq)
        if self.metrics_transition_model is None:
            return JointState.from_position(act_seq)
        goal = self._metrics_goal
        return self.metrics_transition_model.forward(
            self.start_state or self._fallback_start_state(act_seq), act_seq,
            None if goal is None else goal.idxs_current_js,
            None if goal is None else goal.seed_goal_js,
            None if goal is None else goal.idxs_seed_goal_js,
            None if goal is None else goal.seed_enable_implicit_goal_js,
            idxs_env=None if goal is None else goal.idxs_env,
        )

    def compute_state_from_action(self, act_seq, **kwargs):
        del kwargs
        return self._compute_state_from_action_impl(act_seq)

    def compute_state_from_action_metrics(self, act_seq, **kwargs):
        del kwargs
        return self._compute_state_from_action_metrics_impl(act_seq)

    @staticmethod
    def _manager_collection(manager, state, goal, convergence=False, **kwargs):
        if manager is None:
            return CostCollection()
        if convergence:
            return manager.compute_convergence(state, goal=goal, **kwargs)
        return manager.compute_costs(state, goal=goal, **kwargs)

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        return CostsAndConstraints(
            costs=self._manager_collection(self.cost_manager, state, self._num_particles_goal, **kwargs),
            constraints=self._manager_collection(self.constraint_manager, state, self._num_particles_goal, **kwargs),
            hybrid_costs_constraints=self._manager_collection(
                self.hybrid_cost_constraint_manager, state, self._num_particles_goal, **kwargs),
        )

    def _compute_costs_and_constraints_metrics_impl(self, state, **kwargs):
        return CostsAndConstraints(
            costs=self._manager_collection(self.metrics_cost_manager, state, self._metrics_goal, **kwargs),
            constraints=self._manager_collection(self.metrics_constraint_manager, state, self._metrics_goal, **kwargs),
            hybrid_costs_constraints=self._manager_collection(
                self.metrics_hybrid_cost_constraint_manager, state, self._metrics_goal, **kwargs),
        )

    def _compute_convergence_metrics_impl(self, state, **kwargs):
        return self._manager_collection(self.metrics_convergence_manager, state, self._metrics_goal,
                                        convergence=True, **kwargs)

    def evaluate_action(self, act_seq, **kwargs):
        self._validate_action(act_seq)
        self.update_batch_size(act_seq.shape[0])
        state = self._compute_state_from_action_impl(act_seq)
        return RolloutResult(actions=act_seq, state=state,
                             costs_and_constraints=self._compute_costs_and_constraints_impl(state, **kwargs))

    def compute_metrics_from_state(self, state, **kwargs):
        if self._use_cuda_graph:
            # Eager PyTorch is deliberately used instead of silently treating
            # a CPU/MPS execution as a CUDA capture/replay operation.
            pass
        cc = self._compute_costs_and_constraints_metrics_impl(state, **kwargs)
        return RolloutMetrics(state=state, costs_and_constraints=cc,
                              feasible=cc.get_feasible(sum_horizon=self.sum_horizon),
                              convergence=self._compute_convergence_metrics_impl(state, **kwargs))

    def compute_metrics_from_action(self, act_seq, **kwargs):
        self._validate_action(act_seq)
        self.update_batch_size(act_seq.shape[0])
        state = self._compute_state_from_action_metrics_impl(act_seq)
        metrics = self.compute_metrics_from_state(state, **kwargs)
        metrics.actions = act_seq
        return metrics

    def update_params(self, goal: GoalRegistry, num_particles=None):
        if not isinstance(goal, GoalRegistry):
            raise TypeError("goal must be GoalRegistry")
        # A hand-constructed GoalRegistry may carry only a current/seed joint
        # state (no pose or target state from which its dataclass derives the
        # batch size).  Infer that ordinary planning shape before expansion.
        goal.batch_size = self._goal_batch_size(goal)
        if num_particles is not None:
            if isinstance(num_particles, bool) or int(num_particles) < 1:
                raise ValueError("num_particles must be a positive integer")
            self._particle_multiplier = int(num_particles)
        multiplier = self._particle_multiplier
        if goal.current_js is not None:
            self.start_state = (
                goal.current_js.clone()
                if self.start_state is None or not self.start_state._same_shape(goal.current_js)
                else self.start_state.copy_(goal.current_js, allow_clone=False)
            )

        particle_goal = goal.repeat_seeds(multiplier, True) if multiplier is not None else goal.clone()
        particle_layout = self._goal_layout(particle_goal)
        if self._num_particles_goal is None or self._particle_goal_layout != particle_layout:
            # Reallocation is intentional only on shape/topology changes;
            # normal value updates preserve object identity and buffers.
            self._num_particles_goal = particle_goal
            self._particle_goal_layout = particle_layout
        else:
            self._num_particles_goal.copy_(goal, update_idx_buffers=False)
        if self._metrics_goal is None:
            self._metrics_goal = goal.clone()
        else:
            self._metrics_goal.copy_(goal, update_idx_buffers=True)
        if multiplier is not None:
            self.update_batch_size(self._num_particles_goal.batch_size * self._num_particles_goal.num_seeds)
        return True

    def update_goal_dt(self, goal):
        """Update cached seed dt, or accept a scalar for legacy optimizer hooks."""
        if not isinstance(goal, GoalRegistry):
            return self.update_dt(goal)
        if goal.seed_goal_js is None:
            raise ValueError("seed_goal_js is None")
        for cached in (self._num_particles_goal, self._metrics_goal):
            if cached is None:
                raise ValueError("Rollout not initialized. Call update_params first.")
            if cached.seed_goal_js is None:
                raise ValueError("seed_goal_js is None")
            if cached.seed_goal_js.dt is None or goal.seed_goal_js.dt is None or cached.seed_goal_js.dt.shape != goal.seed_goal_js.dt.shape:
                raise ValueError("dt shape mismatch")
            cached.seed_goal_js.dt.copy_(goal.seed_goal_js.dt)
        return True

    def update_batch_size(self, batch_size):
        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self._batch_size == batch_size:
            return
        self._batch_size = batch_size
        for transition in (self.transition_model, self.metrics_transition_model):
            if transition is not None:
                transition.update_batch_size(batch_size)
        for manager in self._cost_manager_list:
            manager.setup_batch_tensors(batch_size, self.horizon)

    def update_dt(self, dt):
        for transition in (self.transition_model, self.metrics_transition_model):
            if transition is not None:
                update = getattr(transition, "update_traj_dt", None)
                if update is None:
                    update = getattr(transition, "update_dt", None)
                if update is None:
                    raise AttributeError("transition model has no dt update method")
                update(dt)
        for manager in self._cost_manager_list:
            manager.update_dt(dt)
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        for manager in self._cost_manager_list:
            manager.reset(reset_problem_ids=reset_problem_ids, **kwargs)
        return True

    def reset_shape(self):
        self._num_particles_goal = None
        self._metrics_goal = None
        self._particle_goal_layout = None
        self._particle_multiplier = None
        return True

    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")

    def reset_seed(self):
        self._sample_generator.manual_seed(int(self.sampler_seed))

    def update_params_cost_managers(self, **kwargs):
        for manager in self._cost_manager_list:
            manager.update_params(**kwargs)

    def enable_cost_component(self, name):
        for manager in self._cost_manager_list:
            if manager.has_cost(name):
                manager.enable_cost_component(name)

    def disable_cost_component(self, name):
        for manager in self._cost_manager_list:
            if manager.has_cost(name):
                manager.disable_cost_component(name)

    def get_cost_component_names(self):
        return [name for manager in self._cost_manager_list for name in manager.get_cost_component_names()]

    def get_all_cost_components(self):
        values = {}
        for manager in self._cost_manager_list:
            values.update(manager.get_cost_components())
        return values

    def get_cost_component_by_name(self, name):
        return [manager.get_cost(name) for manager in self._cost_manager_list if manager.has_cost(name)]

    def filter_robot_state(self, state, *args, **kwargs):
        del args, kwargs
        return state if self.transition_model is None else self.transition_model.filter_robot_state(state)

    def get_robot_command(self, current_state, act_seq, shift_steps=1, state_idx=None, **kwargs):
        if self.transition_model is None:
            raise NotImplementedError("robot command requires a robot state-transition configuration")
        return self.transition_model.get_robot_command(current_state, act_seq, shift_steps, state_idx=state_idx, **kwargs)

    def sample_random_actions(self, n=0, bounded=True, horizon=None, num_samples=None):
        if num_samples is not None:
            n = num_samples
        n = int(n or self.batch_size or 1)
        if n < 0:
            raise ValueError("number of action samples must be non-negative")
        horizon = int(self.action_horizon if horizon is None else horizon)
        if horizon < 0:
            raise ValueError("action horizon must be non-negative")
        if self.device_cfg is None:
            return torch.zeros((n, horizon, self.action_dim))
        values = torch.rand((n, horizon, self.action_dim), generator=self._sample_generator,
                            dtype=self.device_cfg.dtype).to(device=self.device_cfg.device)
        if not bounded or self.transition_model is None:
            return values
        lower, upper = self.action_bound_lows, self.action_bound_highs
        # Bounds are configuration-owned constants.  Third-party transition
        # adapters sometimes construct them on CPU even when the rollout is
        # configured for MPS; normalising *these constants* is safe and keeps
        # generated actions resident with their rollout (unlike moving a
        # caller-supplied action/state inside an evaluation method).
        lower = lower.to(device=values.device, dtype=values.dtype)
        upper = upper.to(device=values.device, dtype=values.dtype)
        finite = torch.isfinite(lower) & torch.isfinite(upper)
        lower = torch.where(finite, lower, torch.zeros_like(lower))
        upper = torch.where(finite, upper, torch.ones_like(upper))
        return lower.view(1, 1, -1) + values * (upper - lower).view(1, 1, -1)

    def get_initial_action(self, use_random=True, use_zero=False, **kwargs):
        del kwargs
        if use_zero:
            if self.device_cfg is None:
                return torch.zeros((self.batch_size or 1, self.action_horizon, self.action_dim))
            return torch.zeros((self.batch_size or 1, self.action_horizon, self.action_dim), **self.device_cfg.as_torch_dict())
        return self.sample_random_actions(self.batch_size or 1) if use_random else None


__all__ = ["RobotRollout", "RobotRolloutCfg"]
