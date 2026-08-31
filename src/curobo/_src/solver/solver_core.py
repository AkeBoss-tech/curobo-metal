"""Shared portable solver infrastructure.

This is deliberately a component rather than a solver base class, matching
cuRoboV2's ownership model.  It owns stable goal/seed state and propagates
updates to any portable rollout objects supplied by a configuration.  CUDA
graphs, streams, and Warp packed-buffer kernels are not emulated: their
explicit methods fail rather than quietly changing execution semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar, Union

import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, create_scene_collision
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.optim.optimizer_protocol import Optimizer
from curobo._src.optim.optim_factory import create_optimizer
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsState
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import is_cuda_graph_reset_available

from .solver_core_cfg import SolverCoreCfg


T_BDOF = TypeVar("T_BDOF", bound=torch.Tensor)


class _SolverCorePortable:
    """Own the portable FK, goal registry, sampling, and rollout lifecycle.

    ``SolverCoreCfg`` can be assembled by a high-level IK/TrajOpt/MPC facade
    without constructing CUDA objects.  Configured rollout instances are
    optional because the production portable solvers compose their own
    objective implementations; callers can also attach normal ``RobotRollout``
    objects and receive the same update/reset propagation.
    """

    def __init__(
        self, config: SolverCoreCfg, scene_collision_checker: Optional[SceneCollision] = None
    ) -> None:
        if not isinstance(config, SolverCoreCfg):
            raise TypeError("config must be SolverCoreCfg")
        self.config = config
        self._scene_collision_checker = scene_collision_checker
        if self._scene_collision_checker is None and config.scene_collision_cfg is not None:
            self._scene_collision_checker = create_scene_collision(config.scene_collision_cfg)

        robot = config.robot_config.kinematics
        self._kinematics = Kinematics(KinematicsCfg(
            config.device_cfg, list(robot.tool_frames), KinematicsParams(robot)
        ))
        joints = [
            joint for joint in robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        lower = config.device_cfg.to_device([joint.limits.lower for joint in joints])
        upper = config.device_cfg.to_device([joint.limits.upper for joint in joints])
        self._goal_manager = GoalManager(config.device_cfg)
        self._seed_manager = SeedManager(
            config.device_cfg, len(joints), lower, upper,
            config.random_seed, action_horizon=1,
        )
        self._goal_buffer = None
        self._solve_state = None
        self._task_initialized = False
        # Bumped for every world replacement.  Eager CPU/MPS callers can use
        # this to invalidate their own shape-keyed objective/scene caches
        # without pretending there is a CUDA graph to reset.
        self._scene_generation = 0
        self._tool_pose_criteria: Dict[str, ToolPoseCriteria] = {
            name: ToolPoseCriteria.disabled() for name in self.tool_frames
        }
        self._joint_position_tracking = False
        # High-level portable solvers generally own their objective/rollout.
        # Still expose the lifecycle collections used by shared applications.
        self.metrics_rollout = None
        self.auxiliary_rollout = None
        self.optimizer_rollouts: List[object] = []
        self.additional_metrics_rollouts: Dict[str, object] = {}
        self.optimizers: List[object] = []
        self._optimizer = None
        self._initialize_configured_components()
        self.attachment_manager = AttachmentManager(
            self._kinematics, self._scene_collision_checker, self.device_cfg
        )
        self.init_state = JointState.from_position(
            self.default_joint_position.unsqueeze(0), self.joint_names
        )

    @staticmethod
    def _is_executable_rollout_config(value: object) -> bool:
        """Return whether a config can build a real eager ``RobotRollout``.

        V2's YAML factory may intentionally carry an uncompiled ``object``
        sentinel until a higher-level facade supplies concrete component
        classes.  Treating that placeholder as a transition model would fail
        at a distant optimization call.  We therefore build direct core
        components only when a typed transition record is present.
        """
        transition = getattr(value, "transition_model_cfg", None)
        return value is not None and transition is not None and hasattr(transition, "robot_config")

    def _initialize_configured_components(self) -> None:
        """Build configured eager rollouts and optimizer stages when possible.

        High-level portable solvers may own their own objective implementation
        and leave these records as uncompiled YAML-shaped transport values.
        Direct users of the V2 ``SolverCoreCfg`` factory, however, expect the
        component to own the fully materialized rollout/optimizer lifecycle.
        This path preserves that useful behavior without fabricating a CUDA
        graph or accepting a raw CUDA/Warp rollout configuration.
        """
        metrics_cfg = self.config.metrics_rollout_config
        if not self._is_executable_rollout_config(metrics_cfg):
            return

        self.metrics_rollout = RobotRollout(metrics_cfg, self._scene_collision_checker)
        self.metrics_rollout.rollout_instance_name = "metrics_rollout"
        self.auxiliary_rollout = RobotRollout(metrics_cfg, self._scene_collision_checker)
        self.auxiliary_rollout.rollout_instance_name = "auxiliary_rollout"

        optimizer_cfgs = list(self.config.optimizer_configs)
        rollout_cfgs = list(self.config.optimizer_rollout_configs)
        if len(optimizer_cfgs) != len(rollout_cfgs):
            raise ValueError(
                "optimizer_configs and optimizer_rollout_configs must have the same length "
                "when metrics_rollout_config is executable"
            )
        for index, (optimizer_cfg, rollout_cfg) in enumerate(zip(optimizer_cfgs, rollout_cfgs)):
            if not self._is_executable_rollout_config(rollout_cfg):
                raise TypeError(
                    f"optimizer_rollout_configs[{index}] must contain a typed transition_model_cfg"
                )
            count = getattr(optimizer_cfg, "num_rollout_instances", None)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"optimizer_configs[{index}].num_rollout_instances must be positive")
            rollouts = [RobotRollout(rollout_cfg, self._scene_collision_checker) for _ in range(count)]
            for subindex, rollout in enumerate(rollouts):
                rollout.rollout_instance_name = f"optimizer_rollout_{index}_{subindex}"
            self.optimizer_rollouts.extend(rollouts)
            self.optimizers.append(create_optimizer(optimizer_cfg, rollouts, use_cuda_graph=False))
        if self.optimizers:
            self._optimizer = MultiStageOptimizer(self.optimizers, self.optimizer_rollouts)

    @property
    def kinematics(self) -> Kinematics:
        return self._kinematics

    @property
    def action_dim(self) -> int:
        return self._kinematics.dof

    @property
    def action_horizon(self) -> int:
        return 1 if self.auxiliary_rollout is None else int(self.auxiliary_rollout.action_horizon)

    @property
    def joint_names(self) -> List[str]:
        return self._kinematics.joint_names

    @property
    def tool_frames(self) -> List[str]:
        return list(self._kinematics.tool_frames)

    @property
    def device_cfg(self):
        return self.config.device_cfg

    @property
    def goal_registry_manager(self) -> GoalManager:
        return self._goal_manager

    @property
    def seed_manager(self) -> SeedManager:
        return self._seed_manager

    @property
    def scene_collision_checker(self) -> Optional[SceneCollision]:
        return self._scene_collision_checker

    @scene_collision_checker.setter
    def scene_collision_checker(self, value: Optional[SceneCollision]) -> None:
        if value is not None and not isinstance(value, SceneCollision):
            raise TypeError("scene_collision_checker must be SceneCollision or None")
        self._scene_collision_checker = value

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def transition_model(self):
        return None if self.auxiliary_rollout is None else self.auxiliary_rollout.transition_model

    @property
    def solve_state(self):
        return self._solve_state

    @property
    def goal_buffer(self):
        """Most recently prepared goal registry, or ``None`` before setup."""
        return self._goal_buffer

    @property
    def task_initialized(self) -> bool:
        """Whether a goal shape has been prepared for this core instance."""
        return self._task_initialized

    @property
    def scene_generation(self) -> int:
        """Monotonic portable-world lifecycle generation."""
        return self._scene_generation

    @property
    def problem_batch_size(self) -> int:
        """Optimizer-facing batch size for the currently prepared problem."""
        if self._solve_state is None:
            return 0
        return self._get_problem_batch_size(self._solve_state)

    @property
    def default_joint_position(self) -> torch.Tensor:
        return self.device_cfg.to_device(self.config.robot_config.kinematics.cspace.default_joint_position)

    @property
    def default_joint_state(self) -> JointState:
        return JointState.from_position(self.default_joint_position, self.joint_names)

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        return self._kinematics.compute_kinematics(state).clone()

    def get_active_js(self, full_js: JointState) -> JointState:
        return self._kinematics.get_active_js(full_js)

    def get_full_js(self, active_js: JointState) -> JointState:
        return self._kinematics.get_full_js(active_js)

    def _structural_goal_change(self, solve_state) -> bool:
        old = self._solve_state
        if old is None:
            return True
        keys = ("solve_type", "batch_size", "num_envs", "num_goalset", "num_seeds",
                "num_ik_seeds", "num_graph_seeds", "num_trajopt_seeds", "tool_frames")
        return any(getattr(old, key, None) != getattr(solve_state, key, None) for key in keys)

    @staticmethod
    def _get_problem_batch_size(solve_state) -> int:
        """Return the seed-expanded number of independently solved problems.

        The native component uses IK seed expansion when present and trajectory
        seed expansion otherwise.  A few high-level portable facades only set
        ``num_seeds``; treating that as a one-step fallback keeps their output
        buffers correctly sized instead of silently selecting a zero-sized
        execution cache.
        """
        if solve_state is None:
            raise TypeError("solve_state must be SolveState")
        batch_size = getattr(solve_state, "batch_size", None)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("solve_state.batch_size must be a positive integer")
        for attribute in ("num_ik_seeds", "num_trajopt_seeds", "num_seeds"):
            seeds = getattr(solve_state, attribute, None)
            if seeds is not None:
                if not isinstance(seeds, int) or seeds < 1:
                    raise ValueError(f"solve_state.{attribute} must be a positive integer")
                return batch_size * seeds
        return batch_size

    def _update_execution_shape(self, problem_batch_size: int) -> None:
        """Synchronize optional eager rollout/optimizer consumers.

        This deliberately uses small duck-typed hooks: the compact portable
        SolverCore can serve production solvers that own their rollouts and
        also direct applications that attach native-style rollout objects.
        """
        for rollout in self.get_all_rollout_instances():
            update = getattr(rollout, "update_batch_size", None)
            if callable(update):
                update(problem_batch_size)
        if self._optimizer is not None:
            update = getattr(self._optimizer, "update_num_problems", None)
            if callable(update):
                update(problem_batch_size)

    def prepare_goal_buffer(
        self, solve_state: SolveState, goal_tool_poses: Optional[GoalToolPose],
        current_state: Optional[JointState] = None, use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None, goal_state: Optional[JointState] = None,
    ):
        """Update and return ``(GoalRegistry, structural_change)``.

        The second item follows V2's contract: it is true when a batch, seed,
        environment, goal-set, mode, or controlled-link shape changed.  Value
        updates reuse the registered shape and only refresh payloads.
        """
        if solve_state is None:
            raise TypeError("solve_state must be SolveState")
        update_reference = self._structural_goal_change(solve_state)
        result = self._goal_manager.update_goal_buffer(
            solve_state,
            goal_tool_poses=goal_tool_poses,
            current_js=current_state,
            seed_goal_js=seed_goal_state,
            goal_js=goal_state,
            use_implicit_goal=use_implicit_goal,
        )
        # GoalManager's older portable API returned a registry alone while
        # current revisions may return the native pair.  Normalize here.
        goal = result[0] if isinstance(result, tuple) else result
        self._solve_state = self._goal_manager.solve_state
        self._goal_buffer = goal
        if update_reference:
            self.reset_shape()
            self._update_execution_shape(self._get_problem_batch_size(solve_state))
            self._task_initialized = True
        # Goal values must reach attached eager rollouts even when their
        # preallocated shape is unchanged.  Native V2 calls this from every
        # solver path; centralizing it here prevents stale pose/state targets
        # for standalone portable SolverCore users.
        self.update_rollout_params(goal)
        return goal, update_reference

    def prepare_action_seeds(
        self, batch_size: int, num_seeds: int, seed_config=None,
        current_state: Optional[JointState] = None, seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._seed_manager.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    def prepare_trajectory_seeds(
        self, batch_size: int, num_seeds: int, current_state: JointState,
        seed_config=None, seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._seed_manager.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config, seed_traj
        )

    def get_all_rollout_instances(
        self, include_optimizer_rollouts: bool = True, include_auxiliary_rollout: bool = True,
    ) -> List[object]:
        values: List[object] = []
        if self.metrics_rollout is not None:
            values.append(self.metrics_rollout)
        if include_auxiliary_rollout and self.auxiliary_rollout is not None:
            values.append(self.auxiliary_rollout)
        values.extend(self.additional_metrics_rollouts.values())
        if include_optimizer_rollouts:
            values.extend(self.optimizer_rollouts)
        return values

    def update_rollout_params(self, goal_buffer, include_auxiliary_rollout: bool = True) -> None:
        if goal_buffer is None:
            raise TypeError("goal_buffer must be GoalRegistry")
        for rollout in self.get_all_rollout_instances(
            include_optimizer_rollouts=False,
            include_auxiliary_rollout=include_auxiliary_rollout,
        ):
            update = getattr(rollout, "update_params", None)
            if callable(update):
                update(goal_buffer)
        if self._optimizer is not None:
            update = getattr(self._optimizer, "update_rollout_params", None)
            if callable(update):
                update(goal_buffer)

    def reset_seed(self) -> None:
        self._seed_manager.reset_seed()
        for rollout in self.get_all_rollout_instances():
            reset = getattr(rollout, "reset_seed", None)
            if callable(reset):
                reset()
        if self._optimizer is not None and hasattr(self._optimizer, "reset_seed"):
            self._optimizer.reset_seed()

    def reset_shape(self) -> None:
        for rollout in self.get_all_rollout_instances():
            reset = getattr(rollout, "reset_shape", None)
            if callable(reset):
                reset()
        if self._optimizer is not None and hasattr(self._optimizer, "reset_shape"):
            self._optimizer.reset_shape()
        self.reset_cuda_graph()

    def reset_cuda_graph(self) -> None:
        """Reset portable persistent execution state.

        Direct CUDA-graph controls are unavailable, but V2 normally calls this
        hook as part of a shape reset.  Treating the eager cache reset as an
        error made ordinary goal updates fail merely because a caller retained
        the upstream ``use_cuda_graph=True`` default.  Explicit capture/debug
        APIs remain unsupported elsewhere.
        """
        if self._optimizer is not None:
            reset = getattr(self._optimizer, "reset_cuda_graph", None)
            if callable(reset):
                reset()
        for rollout in self.get_all_rollout_instances():
            reset = getattr(rollout, "reset_cuda_graph", None)
            if callable(reset):
                try:
                    reset()
                except NotImplementedError:
                    # A portable rollout has no graph; its ordinary shape
                    # reset above is the semantically relevant operation.
                    continue

    def update_world(self, scene_cfg) -> None:
        """Replace the portable collision world and propagate it to rollouts.

        ``SceneCfg``, a per-environment list of ``SceneCfg``, a
        ``SceneCollisionCfg``, or a prebuilt ``SceneCollision`` are supported.
        Asset-path/USD/Warp objects are deliberately rejected rather than
        interpreted as an empty scene.
        """
        from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
        from curobo._src.geom.types import SceneCfg

        if isinstance(scene_cfg, SceneCollision):
            scene = scene_cfg
            cfg = None
        elif isinstance(scene_cfg, SceneCollisionCfg):
            scene = create_scene_collision(scene_cfg)
            cfg = scene_cfg
        elif isinstance(scene_cfg, SceneCfg) or (
            isinstance(scene_cfg, list) and all(isinstance(value, SceneCfg) for value in scene_cfg)
        ):
            cfg = SceneCollisionCfg(
                device_cfg=self.device_cfg,
                scene_model=scene_cfg,
                num_envs=len(scene_cfg) if isinstance(scene_cfg, list) else 1,
            )
            scene = create_scene_collision(cfg)
        else:
            raise NotImplementedError(
                "portable SolverCore world updates require SceneCfg, a list of SceneCfg, "
                "SceneCollisionCfg, or SceneCollision; YAML/USD/Warp assets are unavailable"
            )

        self._scene_collision_checker = scene
        if cfg is not None:
            self.config.scene_collision_cfg = cfg
        for rollout in self.get_all_rollout_instances():
            # The public attribute is enough for eager portable rollouts; a
            # named hook lets richer adapters rebuild their cost managers.
            if hasattr(rollout, "scene_collision_checker"):
                rollout.scene_collision_checker = scene
            update = getattr(rollout, "update_world", None)
            if callable(update):
                update(scene)
        if self._optimizer is not None:
            update = getattr(self._optimizer, "update_world", None)
            if callable(update):
                update(scene)
        self._scene_generation += 1
        # Preserve attached robot-sphere state while replacing the world
        # object against which those spheres are checked.
        self.attachment_manager = AttachmentManager(self._kinematics, scene, self.device_cfg)
        # Scene values, unlike goal values, can invalidate persistent rollout
        # state even when the batch shape did not change.
        self.reset_shape()

    def destroy(self) -> None:
        # No opaque graph/stream handles exist on the portable backend.  Reset
        # Python references so long-running applications can release tensors.
        self.optimizer_rollouts.clear()
        self.additional_metrics_rollouts.clear()
        self.optimizers.clear()
        self._optimizer = None
        self._goal_buffer = None
        self._solve_state = None
        self._task_initialized = False
        self._scene_collision_checker = None
        self.attachment_manager = None

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]) -> None:
        if not isinstance(tool_pose_criteria, dict):
            raise TypeError("tool_pose_criteria must be a mapping")
        invalid = set(tool_pose_criteria).difference(self.tool_frames)
        if invalid:
            raise ValueError(f"unknown tool frame(s): {sorted(invalid)}")
        for name, value in tool_pose_criteria.items():
            if not isinstance(value, ToolPoseCriteria):
                raise TypeError(f"criterion for {name!r} must be ToolPoseCriteria")
            self._tool_pose_criteria[name] = value.clone()
        self.config.tool_pose_criteria = dict(self._tool_pose_criteria)
        for rollout in self.get_all_rollout_instances():
            update = getattr(rollout, "update_params_cost_managers", None)
            if callable(update):
                update(tool_pose_criteria=tool_pose_criteria)

    def enable_tool_pose_tracking(
        self, tool_frames: Optional[List[str]] = None, non_terminal_weight_factor: float = 0.0,
    ) -> None:
        frames = self.tool_frames if tool_frames is None else list(tool_frames)
        self.update_tool_pose_criteria({
            name: ToolPoseCriteria.track_position_and_orientation(
                non_terminal_scale=non_terminal_weight_factor
            ) for name in frames
        })

    def disable_tool_pose_tracking(self, tool_frames: Optional[List[str]] = None) -> None:
        frames = self.tool_frames if tool_frames is None else list(tool_frames)
        self.update_tool_pose_criteria({name: ToolPoseCriteria.disabled() for name in frames})

    def enable_joint_position_tracking(self) -> None:
        self._joint_position_tracking = True
        for rollout in self.get_all_rollout_instances():
            enable = getattr(rollout, "enable_cost_component", None)
            if callable(enable):
                enable("cspace")

    def disable_joint_position_tracking(self) -> None:
        self._joint_position_tracking = False
        for rollout in self.get_all_rollout_instances():
            disable = getattr(rollout, "disable_cost_component", None)
            if callable(disable):
                disable("cspace")

    def sample_configs(
        self, num_samples: int, rejection_ratio: int = 10,
        optimizer_collision_activation_distance: float = 0.01,
    ) -> torch.Tensor:
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        if rejection_ratio < 1:
            raise ValueError("rejection_ratio must be positive")
        if num_samples == 0:
            return torch.empty((0, self.action_dim), **self.device_cfg.as_torch_dict())
        candidates = self._seed_manager.generate_random_actions(
            1, num_samples * rejection_ratio
        ).reshape(-1, self.action_dim)
        if self._scene_collision_checker is None:
            return candidates[:num_samples]
        state = JointState.from_position(candidates[:, None, :], self.joint_names)
        spheres = self.compute_kinematics(state).robot_spheres
        if spheres is None or spheres.shape[-2] == 0:
            return candidates[:num_samples]
        buffer = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
        distance = self._scene_collision_checker.get_sphere_distance_raw(
            spheres, buffer,
            torch.ones((), **self.device_cfg.as_torch_dict()),
            torch.as_tensor(optimizer_collision_activation_distance,
                            **self.device_cfg.as_torch_dict()),
        )
        feasible = distance.amin(dim=(-1, -2)) >= optimizer_collision_activation_distance
        return candidates[feasible][:num_samples]

    def update_link_inertial(
        self, link_name: str, mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None, inertia: Optional[torch.Tensor] = None,
    ) -> None:
        if mass is None and com is None and inertia is None:
            raise ValueError("at least one inertial property must be provided")
        model = self._kinematics._model
        if link_name not in model.link_names:
            raise ValueError(f"unknown link {link_name!r}")
        index = model.link_names.index(link_name)
        if mass is not None:
            if not isinstance(mass, (float, int)) or mass < 0:
                raise ValueError("mass must be a nonnegative scalar")
            model.mass[index] = float(mass)
        if com is not None:
            value = torch.as_tensor(com, **self.device_cfg.as_torch_dict())
            if value.shape != (3,):
                raise ValueError("com must have shape [3]")
            model.com[index].copy_(value)
        if inertia is not None:
            value = torch.as_tensor(inertia, **self.device_cfg.as_torch_dict())
            if value.shape == (6,):
                xx, yy, zz, xy, xz, yz = value
                value = torch.stack((
                    torch.stack((xx, xy, xz)), torch.stack((xy, yy, yz)),
                    torch.stack((xz, yz, zz)),
                ))
            if value.shape != (3, 3):
                raise ValueError("inertia must have shape [3,3] or [6]")
            model.inertia[index].copy_(value)
        # Configured rollouts compile independent robot-model copies.  Keep
        # their inertial state coherent with the core model, just as V2 does
        # when an application updates payload mass after solver construction.
        for rollout in self.get_all_rollout_instances():
            transition = getattr(rollout, "transition_model", None)
            update = getattr(transition, "update_link_inertial", None)
            if callable(update):
                update(link_name, mass, com, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        if not link_properties:
            raise ValueError("link_properties cannot be empty")
        for name, values in link_properties.items():
            unknown = set(values).difference({"mass", "com", "inertia"})
            if unknown:
                raise ValueError(f"unknown inertial properties: {sorted(unknown)}")
            self.update_link_inertial(name, **values)

    def debug_dump(self, file_path: str):
        del file_path
        raise NotImplementedError("CUDA graph debug dumps are unavailable on CPU/MPS")


class SolverCore:
    """Pinned cuRoboV2 declaration surface for the portable solver core."""

    def __init__(
        self, config: SolverCoreCfg, scene_collision_checker: Optional[SceneCollision] = None
    ):
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def kinematics(self) -> Kinematics:
        raise NotImplementedError

    @property
    def transition_model(self):
        raise NotImplementedError

    @property
    def action_horizon(self) -> int:
        raise NotImplementedError

    @property
    def default_joint_position(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def default_joint_state(self) -> JointState:
        raise NotImplementedError

    @property
    def joint_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def tool_frames(self) -> List[str]:
        raise NotImplementedError

    @property
    def solve_state(self) -> SolveState:
        raise NotImplementedError

    @profiler.record_function("solver_core/get_all_rollout_instances")
    def get_all_rollout_instances(
        self, include_optimizer_rollouts: bool = True, include_auxiliary_rollout: bool = True
    ) -> List[RobotRollout]:
        raise NotImplementedError

    @profiler.record_function("solver_core/update_rollout_params")
    def update_rollout_params(
        self, goal_buffer: GoalRegistry, include_auxiliary_rollout: bool = True
    ):
        raise NotImplementedError

    @profiler.record_function("solver_core/reset_shape")
    def reset_shape(self):
        raise NotImplementedError

    @profiler.record_function("solver_core/reset_seed")
    def reset_seed(self):
        raise NotImplementedError

    @profiler.record_function("solver_core/reset_cuda_graph")
    def reset_cuda_graph(self):
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError

    @profiler.record_function("solver_core/prepare_goal_buffer")
    def prepare_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None,
        goal_state: Optional[JointState] = None,
    ):
        raise NotImplementedError

    @profiler.record_function("solver_core/prepare_action_seeds")
    def prepare_action_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        seed_config: Optional[T_BDOF] = None,
        current_state: Optional[JointState] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @profiler.record_function("solver_core/prepare_trajectory_seeds")
    def prepare_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        seed_config: Optional[T_BDOF] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    @profiler.record_function("solver_core/enable_tool_pose_tracking")
    def enable_tool_pose_tracking(
        self, tool_frames: Optional[List[str]] = None, non_terminal_weight_factor: float = 0.0
    ) -> None:
        raise NotImplementedError

    @profiler.record_function("solver_core/disable_tool_pose_tracking")
    def disable_tool_pose_tracking(self, tool_frames: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    @profiler.record_function("solver_core/enable_joint_position_tracking")
    def enable_joint_position_tracking(self) -> None:
        raise NotImplementedError

    @profiler.record_function("solver_core/disable_joint_position_tracking")
    def disable_joint_position_tracking(self) -> None:
        raise NotImplementedError

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        raise NotImplementedError

    @profiler.record_function("solver_core/sample_configs")
    def sample_configs(
        self,
        num_samples: int,
        rejection_ratio: int = 10,
        optimizer_collision_activation_distance: float = 0.01,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        raise NotImplementedError

    def get_active_js(self, full_js: JointState) -> JointState:
        raise NotImplementedError

    def get_full_js(self, active_js: JointState) -> JointState:
        raise NotImplementedError

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]]
    ) -> None:
        raise NotImplementedError

    @profiler.record_function("solver_core/debug_dump")
    def debug_dump(self, file_path: str):
        raise NotImplementedError


# The runtime binding exposes the complete eager CPU/MPS lifecycle, including
# portable-only cache and attachment-management conveniences.
if not TYPE_CHECKING:
    SolverCore = _SolverCorePortable


__all__ = ["SolverCore"]
