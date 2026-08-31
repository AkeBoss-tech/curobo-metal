"""Portable kinematic-tree reduction and joint-state reconstruction.

The upstream implementation rebuilds CUDA packed FK buffers.  Those buffers
are not a useful abstraction on CPU/Metal, so this module instead reduces the
serializable robot tree that feeds the production PyTorch FK backend.  The
result keeps the requested link ancestry, C-space data, joint limits, and a
deterministic record of joints that became fixed for the reduced solve.

Reduction deliberately does *not* compose arbitrary non-zero revolute locks
into a remaining transform.  That requires the CUDA transform-builder's full
tree mutation semantics; callers should leave such a joint active or compose
the transform in their robot configuration before requesting reduction.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Sequence

import torch

from curobo._src.robot.types import CSpaceParams, JointLimits, KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import append_joints_to_state
from curobo._src.util.logging import log_info, log_warn


class KinematicsReducer:
    """Reduce collision-only branches while retaining requested FK outputs.

    The portable representation has one source of truth: ``robot_cfg``.  A
    reduction therefore returns a new :class:`KinematicsParams` built from an
    independent, pruned tree rather than pretending that CUDA link-map tensor
    slices form a standalone kinematic model.
    """

    @classmethod
    def reduce_dof(
        cls,
        original_config: KinematicsParams,
        desired_link_names: List[str],
        remove_collision_spheres: bool = True,
    ) -> KinematicsParams:
        """Return a tree containing only the ancestors of ``desired_link_names``.

        ``desired_link_names`` may select any links in the source tree, not
        only the source's currently stored tool frames.  This is useful when a
        caller has a multi-tool collision model but wants to optimize one
        wrist/link.  The returned tool-frame order is exactly the requested
        order.  Removed active joints are placed in ``lock_jointstate`` at
        their configured retract/default positions, allowing
        :meth:`reconstruct_joint_state` to restore original joint ordering.

        Collision geometry attached to discarded links cannot be represented
        by the reduced FK tree.  It is discarded when
        ``remove_collision_spheres=True``.  Asking to retain it is supported
        only when every sphere already belongs to a retained link; otherwise a
        precise unsupported-boundary error is raised rather than silently
        returning dangling sphere references.
        """
        if not isinstance(original_config, KinematicsParams):
            raise TypeError("original_config must be KinematicsParams")
        desired = cls._validate_desired_links(original_config, desired_link_names)

        source = original_config.robot_cfg
        robot = deepcopy(source)
        source_links = {link.name: link for link in source.links}
        source_joints = {joint.name: joint for joint in source.joints}
        parent_joint = {joint.child: joint for joint in source.joints}

        needed = {source.base_link}
        for link_name in desired:
            current = link_name
            while current not in needed:
                needed.add(current)
                parent = parent_joint.get(current)
                if parent is None:
                    raise ValueError(
                        f"desired link {link_name!r} is not connected to base_link "
                        f"{source.base_link!r}"
                    )
                current = parent.parent

        # Preserve topological order, not input/link-list order.  The tree FK
        # compiler relies on parents preceding children on both CPU and MPS.
        ordered_links = [name for name in original_config.all_link_names if name in needed]
        robot.links = [deepcopy(source_links[name]) for name in ordered_links]
        robot.joints = [
            deepcopy(joint)
            for joint in source.joints
            if joint.parent in needed and joint.child in needed
        ]
        robot.tool_frames = desired.copy()

        selected_joint_names = {joint.name for joint in robot.joints}
        active_names = [
            name for name in original_config.joint_names if name in selected_joint_names
        ]
        if not active_names:
            raise NotImplementedError(
                "portable KinematicsParams currently requires at least one active joint; "
                "fixed-only tree reductions are unsupported"
            )
        cls._validate_retained_mimics(robot.joints, active_names)

        robot.cspace = cls._create_cspace_subset(source.cspace, active_names)
        # ``KinematicsParams.joint_limits`` derives from robot joints, but
        # validating a detached subset here catches source ordering mistakes
        # before constructing a model with unsafe C-space values.
        cls._create_joint_limits_subset(original_config.joint_limits, active_names)

        discarded_sphere_links = {
            sphere.link_name for sphere in source.collision_spheres if sphere.link_name not in needed
        }
        if not remove_collision_spheres and discarded_sphere_links:
            names = sorted(discarded_sphere_links)
            raise NotImplementedError(
                "cannot retain collision spheres attached to discarded links in a "
                f"portable reduced tree: {names}; pass remove_collision_spheres=True"
            )
        robot.collision_spheres = [
            deepcopy(sphere) for sphere in source.collision_spheres if sphere.link_name in needed
        ]
        robot.collision_link_names = [name for name in source.collision_link_names if name in needed]
        robot.self_collision_ignore = cls._filter_link_mapping(source.self_collision_ignore, needed)
        robot.self_collision_buffer = {
            name: value for name, value in source.self_collision_buffer.items() if name in needed
        }

        # A compiled dynamics object describes the original topology, so it
        # must never leak into a reduced tree.  Dynamics can be built lazily
        # from the returned robot if the caller needs it.
        robot.dynamics = None
        robot.metadata = deepcopy(source.metadata)
        robot.metadata["lock_joints"] = cls._build_lock_joints(
            original_config, active_names
        )

        result = KinematicsParams(robot)
        result.make_contiguous()
        return result

    @classmethod
    def reconstruct_joint_state(
        cls,
        reduced_joint_state: JointState,
        lock_jointstate: Optional[JointState],
        target_joint_names: Optional[List[str]] = None,
    ) -> JointState:
        """Combine optimized and locked joints without dropping state channels.

        The operation preserves position, velocity, acceleration, jerk, and
        trajectory metadata.  A supplied lock state is converted to the
        reduced state's device policy, so a CPU configuration may safely
        reconstruct a Metal solver result.  Lock/active name overlap is an
        error: silently choosing one would make the result ambiguous.
        """
        if not isinstance(reduced_joint_state, JointState):
            raise TypeError("reduced_joint_state must be JointState")
        cls._validate_joint_state_names(reduced_joint_state, "reduced_joint_state")
        target = cls._validate_target_names(target_joint_names)

        if lock_jointstate is None or not lock_jointstate.joint_names:
            return (
                reduced_joint_state.clone()
                if target is None
                else reduced_joint_state.reorder(target)
            )
        if not isinstance(lock_jointstate, JointState):
            raise TypeError("lock_jointstate must be JointState or None")
        cls._validate_joint_state_names(lock_jointstate, "lock_jointstate")
        overlap = sorted(set(reduced_joint_state.joint_names).intersection(lock_jointstate.joint_names))
        if overlap:
            raise ValueError(f"reduced and locked states overlap on joints: {overlap}")

        locked = lock_jointstate
        if locked.device_cfg != reduced_joint_state.device_cfg:
            locked = locked.to(reduced_joint_state.device_cfg)
        # Static lock records created by reduce_dof have all derivative
        # channels.  For a caller-provided partial record, fill absent
        # channels with zeros so an optimized velocity/acceleration is not
        # accidentally erased by state concatenation.
        locked = cls._materialize_lock_channels(locked)
        locked = cls._broadcast_lock_to_state(locked, reduced_joint_state)
        full = append_joints_to_state(reduced_joint_state, locked)
        return full if target is None else full.reorder(target)

    @staticmethod
    def _create_joint_limits_subset(
        original_limits: JointLimits, joint_names: List[str]
    ) -> JointLimits:
        """Return named joint-limit columns on the original CPU/MPS device."""
        if not isinstance(original_limits, JointLimits):
            raise TypeError("original_limits must be JointLimits")
        return original_limits.reindex(list(joint_names))

    @staticmethod
    def _create_cspace_subset(
        original_cspace: CSpaceParams | Any, joint_names: List[str]) -> Any:
        """Copy/reorder tensor ``CSpaceParams`` or serializable robot C-space.

        The high-level portable robot model intentionally stores YAML-friendly
        lists, whereas some direct internal callers provide the upstream-style
        tensor :class:`CSpaceParams`.  Supporting both makes reduction useful
        across public config loading and lower-level solver assembly.
        """
        names = list(joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be a non-empty unique sequence")
        if isinstance(original_cspace, CSpaceParams):
            result = original_cspace.clone()
            result.inplace_reindex(names)
            return result
        if not hasattr(original_cspace, "joint_names"):
            raise TypeError("original_cspace must expose joint_names")
        source_names = list(original_cspace.joint_names)
        unknown = [name for name in names if name not in source_names]
        if unknown:
            raise ValueError(f"joint_names are absent from original C-space: {unknown}")
        result = deepcopy(original_cspace)
        indices = [source_names.index(name) for name in names]
        result.joint_names = names
        # These are the portable serializable per-DOF fields.  Preserve an
        # omitted field as omitted; arrays/tensors must have exact source DOF
        # width or the source configuration itself is malformed.
        for field in (
            "default_joint_position", "cspace_distance_weight", "null_space_weight",
            "null_space_maximum_distance", "max_velocity", "max_acceleration", "max_jerk",
            "velocity_scale", "acceleration_scale", "jerk_scale", "position_limit_clip",
        ):
            if not hasattr(result, field):
                continue
            value = getattr(result, field)
            setattr(result, field, KinematicsReducer._subset_cspace_value(
                value, indices, len(source_names), field
            ))
        return result

    @staticmethod
    def _validate_desired_links(config: KinematicsParams, desired_link_names: Sequence[str]) -> list[str]:
        if not isinstance(desired_link_names, (list, tuple)):
            raise TypeError("desired_link_names must be a list or tuple of link names")
        desired = list(desired_link_names)
        if not desired:
            raise ValueError("desired_link_names must not be empty")
        if any(not isinstance(name, str) or not name for name in desired):
            raise TypeError("desired_link_names must contain non-empty strings")
        if len(set(desired)) != len(desired):
            raise ValueError("desired_link_names must not contain duplicates")
        known = set(config.all_link_names)
        missing = [name for name in desired if name not in known]
        if missing:
            raise ValueError(
                f"None of the desired links {desired} found in the kinematic tree; "
                f"unknown: {missing}"
            )
        return desired

    @staticmethod
    def _validate_retained_mimics(joints: Sequence[Any], active_names: Sequence[str]) -> None:
        active = set(active_names)
        for joint in joints:
            source = getattr(joint, "mimic_joint", None)
            if source is not None and source not in active:
                raise NotImplementedError(
                    f"reduced mimic joint {joint.name!r} depends on non-retained active joint "
                    f"{source!r}; compose/lock that transform before reduction"
                )

    @staticmethod
    def _filter_link_mapping(values: dict[str, list[str]], needed: set[str]) -> dict[str, list[str]]:
        return {
            name: [other for other in others if other in needed]
            for name, others in values.items()
            if name in needed
        }

    @staticmethod
    def _subset_cspace_value(value: Any, indices: list[int], source_dof: int, field: str) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.clone()
            if value.ndim != 1 or value.numel() < source_dof:
                raise ValueError(f"{field} must be scalar or contain at least one value per source joint")
            index = torch.tensor(indices, dtype=torch.long, device=value.device)
            return value.index_select(0, index).clone()
        if isinstance(value, (list, tuple)):
            if len(value) < source_dof:
                raise ValueError(f"{field} must contain at least one value per source joint")
            subset = [value[index] for index in indices]
            return tuple(subset) if isinstance(value, tuple) else subset
        # Scalar limits/scales describe all DOFs and are valid unchanged.
        return deepcopy(value)

    @classmethod
    def _build_lock_joints(
        cls, original_config: KinematicsParams, active_names: Sequence[str]
    ) -> dict[str, float]:
        source = original_config.robot_cfg
        active = set(active_names)
        existing = source.metadata.get("lock_joints", {})
        if existing is None:
            existing = {}
        if not isinstance(existing, dict):
            raise TypeError("robot metadata lock_joints must be a mapping")
        locks: dict[str, float] = {}
        for name, value in existing.items():
            if name not in active:
                locks[str(name)] = cls._as_scalar_lock_value(value, str(name))

        defaults = cls._default_positions(source.cspace, original_config.joint_names)
        for name in original_config.joint_names:
            if name not in active and name not in locks:
                locks[name] = defaults[name]
        return locks

    @staticmethod
    def _default_positions(cspace: Any, joint_names: Sequence[str]) -> dict[str, float]:
        raw = getattr(cspace, "default_joint_position", None)
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            return {name: 0.0 for name in joint_names}
        if isinstance(raw, torch.Tensor):
            if raw.ndim != 1 or raw.numel() != len(joint_names):
                raise ValueError("default_joint_position must be a rank-1 source-DOF vector")
            raw = raw.detach().cpu().tolist()
        if not isinstance(raw, (list, tuple)) or len(raw) != len(joint_names):
            raise ValueError("default_joint_position must contain one value per source joint")
        return {
            name: KinematicsReducer._as_scalar_lock_value(value, name)
            for name, value in zip(joint_names, raw)
        }

    @staticmethod
    def _as_scalar_lock_value(value: Any, name: str) -> float:
        tensor = torch.as_tensor(value)
        if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"lock value for joint {name!r} must be one finite scalar")
        return float(tensor.item())

    @staticmethod
    def _validate_joint_state_names(state: JointState, label: str) -> None:
        names = state.joint_names
        if names is None:
            raise ValueError(f"{label}.joint_names is required")
        if len(names) != state.position.shape[-1]:
            raise ValueError(f"{label}.joint_names must match its final position dimension")
        if len(set(names)) != len(names):
            raise ValueError(f"{label}.joint_names must be unique")

    @staticmethod
    def _validate_target_names(target_joint_names: Optional[Sequence[str]]) -> Optional[list[str]]:
        if target_joint_names is None:
            return None
        target = list(target_joint_names)
        if not target or any(not isinstance(name, str) or not name for name in target):
            raise ValueError("target_joint_names must be a non-empty sequence of names")
        if len(set(target)) != len(target):
            raise ValueError("target_joint_names must be unique")
        return target

    @staticmethod
    def _materialize_lock_channels(lock: JointState) -> JointState:
        channels = {}
        for name in ("velocity", "acceleration", "jerk"):
            value = getattr(lock, name)
            if value is not None:
                channels[name] = value
            else:
                channels[name] = torch.zeros_like(lock.position)
        return JointState(
            lock.position, channels["velocity"], channels["acceleration"],
            lock.joint_names.copy(), channels["jerk"], lock.device_cfg,
            lock.dt, aux_data=dict(lock.aux_data), knot=lock.knot,
            knot_dt=lock.knot_dt, control_space=lock.control_space,
        )

    @staticmethod
    def _broadcast_lock_to_state(lock: JointState, state: JointState) -> JointState:
        """Broadcast static/row-wise locks across a reduced state prefix.

        ``lock_jointstate`` is normally a rank-one configuration vector while
        an optimizer emits ``[batch, dof]`` or ``[batch, horizon, dof]``.  The
        CUDA implementation's packed append broadcasts this implicitly.  Do
        it openly here so the ordinary PyTorch concatenation retains that
        practical behavior on CPU and MPS.
        """
        target_shape = (*state.position.shape[:-1], lock.position.shape[-1])

        def expand(value: torch.Tensor) -> torch.Tensor:
            if value.shape[-1] != lock.position.shape[-1]:
                raise ValueError("lock-state channels must share its final joint dimension")
            if value.ndim > len(target_shape):
                raise ValueError("lock-state rank cannot exceed reduced-state rank")
            padded = value.reshape((1,) * (len(target_shape) - value.ndim) + tuple(value.shape))
            try:
                return padded.expand(target_shape)
            except RuntimeError as error:
                raise ValueError(
                    "lock-state batch/horizon dimensions are not broadcastable to reduced state"
                ) from error

        return JointState(
            expand(lock.position), expand(lock.velocity), expand(lock.acceleration),
            lock.joint_names.copy(), expand(lock.jerk), state.device_cfg,
            # Timing belongs to the optimized state returned by cat_joint_states;
            # retaining lock timing here only complicates rank validation.
            None, aux_data=dict(lock.aux_data), control_space=lock.control_space,
        )


__all__ = ["KinematicsReducer"]
