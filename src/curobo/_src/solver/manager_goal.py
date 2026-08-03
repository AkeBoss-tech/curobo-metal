"""Portable goal-buffer lifecycle manager.

The native implementation owns shape-keyed goal buffers so repeated calls can
update values without reallocating CUDA graph inputs.  This implementation
keeps that observable lifecycle on CPU and MPS using ordinary PyTorch tensors;
it deliberately does *not* emulate CUDA graph handles or raw packed-buffer
ABIs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


class GoalManager:
    """Create, cache, select, and update seed-expanded solver goals.

    A returned ``reference_updated`` flag means that callers must refresh any
    shape-keyed execution cache.  A smaller goal set is the one intentional
    exception: it is padded to a previously allocated larger set and therefore
    leaves the reference stable, matching cuRobo V2's goal-set cache policy.
    """

    def __init__(self, device_cfg: DeviceCfg):
        self.device_cfg = device_cfg
        self._goal_buffer: Optional[GoalRegistry] = None
        self._solve_state = None
        self._col: Optional[torch.Tensor] = None

    @staticmethod
    def _require_positive_int(value, name: str) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _require_solve_state(self, solve_state) -> None:
        if solve_state is None:
            raise TypeError("solve_state must be provided")
        self._require_positive_int(getattr(solve_state, "batch_size", None), "solve_state.batch_size")
        self._require_positive_int(getattr(solve_state, "num_goalset", None), "solve_state.num_goalset")

    @staticmethod
    def _num_seeds(solve_state) -> int:
        """Use V2's single-candidate default for a sparse generic SolveState."""
        seeds = getattr(solve_state, "num_seeds", None)
        return 1 if seeds is None else GoalManager._require_positive_int(seeds, "solve_state.num_seeds")

    def _validate_device(self, value, name: str) -> None:
        """Reject hidden cross-device copies at the portable solver boundary.

        ``JointState`` and ``GoalToolPose`` are compound values.  Checking only
        their leading position tensor would let a caller construct a mixed
        device payload which then fails (or, worse, triggers an implicit copy)
        in an unrelated rollout operation.  Validate every materialized tensor
        at this public boundary instead.
        """
        if value is None:
            return
        if isinstance(value, JointState):
            tensors = [
                (field, getattr(value, field))
                for field in ("position", "velocity", "acceleration", "jerk", "dt", "knot", "knot_dt")
                if getattr(value, field) is not None
            ]
        elif isinstance(value, GoalToolPose):
            tensors = [("position", value.position), ("quaternion", value.quaternion)]
        elif isinstance(value, torch.Tensor):
            tensors = [("tensor", value)]
        else:
            raise TypeError(f"{name} must be a tensor or a supported cuRobo value type")
        for field, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name}.{field} must be a torch.Tensor")
            if not self.device_cfg.is_same_torch_device(tensor.device):
                raise ValueError(
                    f"{name}.{field} is on {tensor.device}, expected {self.device_cfg.device}; "
                    "move inputs explicitly instead of relying on an implicit host copy"
                )

    def _validate_inputs(
        self,
        solve_state,
        goal_tool_poses: Optional[GoalToolPose],
        goal_js: Optional[JointState],
        current_js: Optional[JointState],
        seed_goal_js: Optional[JointState],
        current_state_dt: Optional[torch.Tensor],
    ) -> None:
        self._validate_device(goal_tool_poses, "goal_tool_poses")
        self._validate_device(goal_js, "goal_js")
        self._validate_device(current_js, "current_js")
        self._validate_device(seed_goal_js, "seed_goal_js")
        self._validate_device(current_state_dt, "current_state_dt")
        if goal_tool_poses is not None:
            if not goal_tool_poses.tool_frames or len(set(goal_tool_poses.tool_frames)) != len(goal_tool_poses.tool_frames):
                raise ValueError("goal_tool_poses.tool_frames must be a non-empty unique sequence")
            if goal_tool_poses.batch_size != solve_state.batch_size:
                raise ValueError("goal_tool_poses batch size must match solve_state.batch_size")
            if goal_tool_poses.num_goalset != solve_state.num_goalset:
                raise ValueError("goal_tool_poses goalset size must match solve_state.num_goalset")
        for name, state in (("goal_js", goal_js), ("current_js", current_js)):
            if state is not None and (state.position.ndim < 2 or state.position.shape[0] != solve_state.batch_size):
                raise ValueError(f"{name} batch size must match solve_state.batch_size")
        if seed_goal_js is not None and seed_goal_js.position.ndim != 3:
            raise ValueError("seed_goal_js must have shape [batch, seeds, dof]")
        if seed_goal_js is not None and seed_goal_js.position.shape[0] != solve_state.batch_size:
            raise ValueError("seed_goal_js batch size must match solve_state.batch_size")
        if current_state_dt is not None:
            if current_state_dt.ndim > 2 or (
                current_state_dt.ndim > 0 and current_state_dt.shape[0] not in (1, solve_state.batch_size)
            ):
                raise ValueError("current_state_dt must be scalar, [1], [batch], or [batch, 1]")

    def create_goal_buffer(
        self,
        solve_state,
        goal_tool_poses: Optional[GoalToolPose] = None,
        goal_js: Optional[JointState] = None,
        current_js: Optional[JointState] = None,
        seed_goal_js: Optional[JointState] = None,
        current_state_dt: Optional[torch.Tensor] = None,
    ) -> GoalRegistry:
        """Create a seed-expanded registry without storing it as the active one."""
        self._require_solve_state(solve_state)
        self._validate_inputs(solve_state, goal_tool_poses, goal_js, current_js, seed_goal_js, current_state_dt)
        registry = GoalRegistry.create_idx(
            pose_batch_size=solve_state.batch_size,
            multi_env=solve_state.multi_env,
            num_seeds=self._num_seeds(solve_state),
            device_cfg=self.device_cfg,
            seed_goal_state=seed_goal_js,
            repeat_seed_idx_buffers=False,
        )
        registry.goal_js = goal_js
        registry.link_goal_poses = goal_tool_poses
        registry.current_js = current_js
        registry.current_state_dt = (
            current_state_dt
            if current_state_dt is not None
            else (None if current_js is None else current_js.dt)
        )
        # The compact portable MPC facade historically calls this constructor
        # during ``setup`` and then updates through the manager.  Retaining the
        # created registry here gives that normal lifecycle a stable active
        # buffer while remaining harmless for callers that immediately invoke
        # ``update_goal_buffer``.
        self._solve_state = solve_state
        self._goal_buffer = registry
        self.update_batch_helper(solve_state.batch_size)
        return registry

    @staticmethod
    def _copy_joint_state_values(target: JointState, source: JointState, label: str) -> None:
        """Copy materialized state channels without replacing a cached object.

        High-level APIs are allowed to provide a position-only state after a
        setup call whose cached state has timing but no velocity (or vice
        versa).  Position is the structural channel; optional derivative and
        timing channels are copied only when both buffers materialize them.
        """
        if target.position.shape != source.position.shape:
            raise ValueError(f"{label} shape does not match the existing goal buffer")
        target.position.copy_(source.position)
        for field in ("velocity", "acceleration", "jerk", "dt", "knot", "knot_dt"):
            destination, value = getattr(target, field), getattr(source, field)
            if destination is not None and value is not None:
                if destination.shape != value.shape:
                    raise ValueError(f"{label} {field} shape does not match the existing goal buffer")
                destination.copy_(value)
        if source.joint_names is not None:
            target.joint_names = source.joint_names.copy()

    @staticmethod
    def _new_payload_requires_reference(current: GoalRegistry, **values) -> bool:
        """Whether a requested payload has no preallocated destination."""
        for field, value in values.items():
            if value is not None and getattr(current, field) is None:
                return True
        return False

    def update_goal_buffer(
        self,
        solve_state,
        goal_tool_poses: Optional[GoalToolPose] = None,
        current_js: Optional[JointState] = None,
        seed_goal_js: Optional[JointState] = None,
        goal_js: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        current_state_dt: Optional[torch.Tensor] = None,
    ) -> Tuple[GoalRegistry, bool]:
        """Update the active registry and report whether its backing shape changed."""
        self._require_solve_state(solve_state)
        self._validate_inputs(solve_state, goal_tool_poses, goal_js, current_js, seed_goal_js, current_state_dt)
        update_reference = self._goal_buffer is None or self._solve_state is None
        if not update_reference:
            update_reference = self._new_payload_requires_reference(
                self._goal_buffer,
                goal_js=goal_js,
                link_goal_poses=goal_tool_poses,
                current_js=current_js,
                seed_goal_js=seed_goal_js,
            )

        if not update_reference and self._solve_state != solve_state:
            padded = self._get_padded_goalset_for_links(
                solve_state, self._solve_state, self._goal_buffer, goal_tool_poses
            )
            if padded is None:
                update_reference = True
            else:
                goal_tool_poses = padded

        if update_reference:
            self._solve_state = solve_state
            self._goal_buffer = self.create_goal_buffer(
                solve_state, goal_tool_poses, goal_js, current_js, seed_goal_js, current_state_dt
            )
            self.update_batch_helper(solve_state.batch_size)
        else:
            assert self._goal_buffer is not None
            if current_js is not None:
                self._copy_joint_state_values(self._goal_buffer.current_js, current_js, "current state")
            if goal_js is not None:
                self._copy_joint_state_values(self._goal_buffer.goal_js, goal_js, "goal state")
            if goal_tool_poses is not None:
                # Link names and full tensor shape were checked by padding or
                # by the stable reference condition below; never replace a
                # caller-visible preallocated object on a value-only update.
                stored = self._goal_buffer.link_goal_poses
                if stored is None or set(stored.tool_frames) != set(goal_tool_poses.tool_frames):
                    raise ValueError("goal link poses do not match the existing goal buffer")
                if stored.position.shape != goal_tool_poses.position.shape or stored.quaternion.shape != goal_tool_poses.quaternion.shape:
                    raise ValueError("goal link pose shape does not match the existing goal buffer")
                stored.position.copy_(goal_tool_poses.position)
                stored.quaternion.copy_(goal_tool_poses.quaternion)
            if seed_goal_js is not None:
                self._goal_buffer.seed_goal_js.copy_(seed_goal_js, allow_clone=False)
            effective_dt = current_state_dt
            if effective_dt is None and current_js is not None:
                effective_dt = current_js.dt
            if effective_dt is not None:
                target = self._goal_buffer.current_state_dt
                if target is None:
                    self._goal_buffer.current_state_dt = effective_dt.clone()
                elif target.shape != effective_dt.shape:
                    raise ValueError("current_state_dt shape does not match the existing goal buffer")
                else:
                    target.copy_(effective_dt)

        if use_implicit_goal and self._goal_buffer.seed_enable_implicit_goal_js is not None:
            self._goal_buffer.seed_enable_implicit_goal_js.fill_(1)
        elif self._goal_buffer.seed_enable_implicit_goal_js is not None:
            self._goal_buffer.seed_enable_implicit_goal_js.zero_()
        return self._goal_buffer, update_reference

    def update_from_goal_registry(self, solve_state, goal: GoalRegistry) -> Tuple[GoalRegistry, bool]:
        """Install/update a rollout-owned registry while preserving backing buffers."""
        self._require_solve_state(solve_state)
        if not isinstance(goal, GoalRegistry):
            raise TypeError("goal must be a GoalRegistry")
        self._validate_inputs(
            solve_state,
            goal.link_goal_poses, goal.goal_js, goal.current_js,
            goal.seed_goal_js, goal.current_state_dt,
        )
        update_reference = self._goal_buffer is None or self._solve_state is None
        if not update_reference:
            current = self._goal_buffer
            # A missing state channel is structural here: a rollout registry
            # represents its whole requested problem rather than a sparse
            # value patch like ``update_goal_buffer``.
            for field in ("goal_js", "seed_goal_js"):
                if (getattr(current, field) is None) != (getattr(goal, field) is None):
                    update_reference = True
                    break
            if not update_reference and self._solve_state != solve_state:
                padded = self._get_padded_goalset_for_links(
                    solve_state, self._solve_state, current, goal.link_goal_poses
                )
                if padded is None:
                    update_reference = True
                else:
                    goal = goal.clone()
                    goal.link_goal_poses = padded

        if update_reference:
            self._solve_state = solve_state
            self._goal_buffer = goal.create_index_buffers(
                solve_state.batch_size, solve_state.multi_env, self._num_seeds(solve_state), self.device_cfg
            )
            if goal.idxs_seed_goal_js is not None:
                self._goal_buffer.idxs_seed_goal_js = goal.idxs_seed_goal_js.clone()
            if goal.seed_enable_implicit_goal_js is not None:
                self._goal_buffer.seed_enable_implicit_goal_js = goal.seed_enable_implicit_goal_js.clone()
            self.update_batch_helper(solve_state.batch_size)
        else:
            self._goal_buffer.copy_(goal, update_idx_buffers=False, allow_clone=False)
        return self._goal_buffer, update_reference

    def update_batch_helper(self, batch_size: int) -> torch.Tensor:
        self._require_positive_int(batch_size, "batch_size")
        self._col = torch.arange(batch_size, device=self.device_cfg.device, dtype=torch.long).view(-1, 1)
        return self._col

    def _require_initialized(self) -> GoalRegistry:
        if self._goal_buffer is None or self._solve_state is None:
            raise RuntimeError("goal buffer has not been initialized")
        return self._goal_buffer

    def update_goal_tool_poses(self, goal_tool_poses: GoalToolPose) -> GoalRegistry:
        buffer = self._require_initialized()
        self._validate_device(goal_tool_poses, "goal_tool_poses")
        stored = buffer.link_goal_poses
        if stored is None or set(goal_tool_poses.tool_frames) != set(stored.tool_frames):
            raise ValueError("goal link poses do not match the existing goal buffer")
        if goal_tool_poses.position.shape != stored.position.shape or goal_tool_poses.quaternion.shape != stored.quaternion.shape:
            raise ValueError("goal link pose shape does not match the existing goal buffer")
        updated, changed = self.update_goal_buffer(self._solve_state, goal_tool_poses=goal_tool_poses)
        if changed:
            raise RuntimeError("goal link poses unexpectedly changed the goal buffer reference")
        return updated

    def update_current_state(self, current_state: JointState) -> GoalRegistry:
        buffer = self._require_initialized()
        self._validate_device(current_state, "current_state")
        if buffer.current_js is None or current_state.shape != buffer.current_js.shape:
            raise ValueError("current state does not match the existing goal buffer")
        updated, changed = self.update_goal_buffer(self._solve_state, current_js=current_state)
        if changed:
            raise RuntimeError("current state unexpectedly changed the goal buffer reference")
        return updated

    def update_goal_state(self, goal_state: JointState) -> GoalRegistry:
        buffer = self._require_initialized()
        self._validate_device(goal_state, "goal_state")
        if buffer.goal_js is None or goal_state.shape != buffer.goal_js.shape:
            raise ValueError("goal state does not match the existing goal buffer")
        updated, changed = self.update_goal_buffer(self._solve_state, goal_js=goal_state)
        if changed:
            raise RuntimeError("goal state unexpectedly changed the goal buffer reference")
        return updated

    @property
    def goal_buffer(self) -> GoalRegistry:
        return self._require_initialized()

    @property
    def solve_state(self):
        self._require_initialized()
        return self._solve_state

    @property
    def batch_helper(self) -> torch.Tensor:
        if self._col is None:
            raise RuntimeError("batch helper has not been initialized")
        return self._col.clone()

    def get_batch_size(self) -> int:
        return 0 if self._solve_state is None else self._solve_state.get_batch_size()

    def get_ik_batch_size(self) -> int:
        return 0 if self._solve_state is None else self._solve_state.get_ik_batch_size()

    def get_trajopt_batch_size(self) -> int:
        return 0 if self._solve_state is None else self._solve_state.get_trajopt_batch_size()

    @staticmethod
    def _get_padded_goalset_for_links(solve_state, current_solve_state, current_goal_buffer, links_goal_pose):
        """Pad a smaller link-goal set so a larger cached registry remains valid."""
        stored = current_goal_buffer.link_goal_poses
        if links_goal_pose is None or stored is None:
            return None
        if solve_state.solve_type != current_solve_state.solve_type:
            return None
        if solve_state.batch_size != current_solve_state.batch_size:
            return None
        if set(links_goal_pose.tool_frames) != set(stored.tool_frames):
            return None
        old_count, new_count = current_solve_state.num_goalset, solve_state.num_goalset
        if new_count > old_count:
            return None
        if links_goal_pose.position.shape[0] != stored.position.shape[0]:
            return None
        # Every rank other than the goal-set axis must already agree.  This
        # protects cached batched/horizon/link layouts from accidental broadcast.
        if (
            links_goal_pose.position.shape[:3] != stored.position.shape[:3]
            or links_goal_pose.position.shape[-1] != stored.position.shape[-1]
            or links_goal_pose.quaternion.shape[:3] != stored.quaternion.shape[:3]
            or links_goal_pose.quaternion.shape[-1] != stored.quaternion.shape[-1]
            or links_goal_pose.position.shape[3] != new_count
            or links_goal_pose.quaternion.shape[3] != new_count
        ):
            return None
        padded = stored.clone()
        padded.position[:, :, :, :new_count, :].copy_(links_goal_pose.position)
        padded.quaternion[:, :, :, :new_count, :].copy_(links_goal_pose.quaternion)
        if new_count < old_count:
            padded.position[:, :, :, new_count:, :].copy_(links_goal_pose.position[:, :, :, :1, :])
            padded.quaternion[:, :, :, new_count:, :].copy_(links_goal_pose.quaternion[:, :, :, :1, :])
        return padded


__all__ = ["GoalManager"]
