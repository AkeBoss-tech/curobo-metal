"""Differentiable portable rigid-body inverse dynamics."""

from __future__ import annotations

from typing import Optional, Union
from copy import deepcopy

import numpy as np
import torch

from curobo._src.state.state_joint import JointState
from curobo_metal.ops.whole_body import (
    WholeBodyState,
    forward_dynamics,
    mass_matrix,
    rollout_dynamics,
    tree_forward_kinematics,
)

from .dynamics_cfg import DynamicsCfg


def _compute_threads_per_batch(max_level_width: int) -> int:
    """Return the pinned CUDA heuristic without claiming CUDA execution.

    Some applications use this helper when sizing their own compatibility
    buffers.  The portable backend does not launch a tree-parallel CUDA
    kernel, but the deterministic sizing policy remains useful metadata.
    """
    if max_level_width < 0:
        raise ValueError("max_level_width must be nonnegative")
    if max_level_width < 4:
        return 1
    threads = 1
    while threads < max_level_width:
        threads *= 2
    return min(threads, 32)


class Dynamics:
    """Differentiable CPU/MPS rigid-body inverse dynamics.

    This is a production PyTorch implementation over the portable tree RNEA
    operator.  It preserves the cuRobo lifecycle and tensor ranks, including
    optional per-link external wrenches.  CUDA packed-buffer kernels and their
    ABI are intentionally not emulated.
    """

    def __init__(self, config: DynamicsCfg) -> None:
        self.config = config
        self.kinematics_config = config.kinematics_config
        self.device_cfg = config.device_cfg
        self.dof = config.kinematics_config.num_dof
        self.device = config.device_cfg.device.type
        robot_cfg = deepcopy(self.kinematics_config.robot_cfg)
        # Dynamics in pinned V2 consumes ``KinematicsParams``' packed
        # inertial tensors, rather than reparsing the authored URDF at launch.
        # Materialize those values into the portable tree before compiling
        # RNEA so loader-compatible mass/CoM/default-inertia semantics are
        # observed by CPU and MPS alike.
        inertial_tree = self.kinematics_config._tree()
        masses_com = self.kinematics_config.link_masses_com.detach().cpu()
        inertias = self.kinematics_config.link_inertias.detach().cpu()
        links_by_name = {link.name: link for link in robot_cfg.links}
        for index, tree_link in enumerate(inertial_tree.links):
            link = links_by_name[tree_link.name]
            packed = masses_com[index]
            link.com = tuple(float(value) for value in packed[:3].tolist())
            link.mass = float(packed[3].item())
            inertia = inertias[index, :6]
            link.inertia = tuple(float(value) for value in inertia.tolist())
        # Several standard cuRobo URDF assets contain inertial coefficients
        # that are not positive semidefinite. The CUDA implementation accepts
        # them; project onto the physical PSD cone before compiling RNEA.
        for link in robot_cfg.links:
            xx, yy, zz, xy, xz, yz = link.inertia
            matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            values, vectors = np.linalg.eigh(matrix)
            if values.min() < 0:
                matrix = (vectors * np.maximum(values, 0.0)) @ vectors.T
                link.inertia = (
                    matrix[0, 0], matrix[1, 1], matrix[2, 2],
                    matrix[0, 1], matrix[0, 2], matrix[1, 2],
                )
        self._model = robot_cfg.to_whole_body_model(
            device=config.device_cfg.device, dtype=config.device_cfg.dtype
        )
        self._model.gravity.copy_(config.get_gravity())
        self._backend_joint_names = tuple(self._model.joint_names)
        self._joint_reorder_indices: torch.Tensor | None = None
        self._joint_reorder_source: tuple[str, ...] | None = None
        self._n_links = self._model.link_count
        self._n_dof = self._model.dof
        self._gravity_spatial = config.get_gravity_spatial()
        # Keep the source-visible metadata aliases.  They are ordinary
        # portable tensors, not arguments for a raw CUDA packed-buffer launch,
        # but consumers use them for introspection and inertial updates.
        kp = self.kinematics_config
        self._fixed_transforms = kp.fixed_transforms
        self._link_masses_com = kp.link_masses_com
        self._link_inertias = kp.link_inertias
        self._joint_map_type = kp.joint_map_type
        self._joint_map = kp.joint_map
        self._link_map = kp.link_map
        self._joint_offset_map = kp.joint_offset_map
        self._level_starts = kp.link_level_offsets
        self._level_links = kp.link_level_data
        self._n_levels = kp.n_tree_levels
        self._threads_per_batch = _compute_threads_per_batch(
            self.kinematics_config.max_level_width
        )
        self.batch_size: int | None = None
        self.horizon: int | None = None
        self._batch_size: int | None = None
        self._horizon: int | None = None
        self._total_batch: int | None = None
        self._tau_buffer: torch.Tensor | None = None
        self._grad_q_buffer: torch.Tensor | None = None
        self._grad_qd_buffer: torch.Tensor | None = None
        self._grad_qdd_buffer: torch.Tensor | None = None
        self._grad_f_ext_buffer: torch.Tensor | None = None
        self._forward_cache: torch.Tensor | None = None

    def setup_batch_size(self, batch_size: int, horizon: int = 1) -> None:
        if not isinstance(batch_size, int) or not isinstance(horizon, int):
            raise TypeError("batch_size and horizon must be integers")
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        self.batch_size = self._batch_size = batch_size
        self.horizon = self._horizon = horizon
        self._allocate_buffers(batch_size * horizon)

    def _allocate_buffers(self, total_batch: int) -> None:
        """Allocate compatibility buffers without routing autograd through them."""
        if total_batch < 1:
            raise ValueError("total_batch must be positive")
        shape = (total_batch, self._n_dof)
        options = {"device": self._model.device, "dtype": self._model.dtype}
        self._tau_buffer = torch.zeros(shape, **options)
        self._grad_q_buffer = torch.zeros(shape, **options)
        self._grad_qd_buffer = torch.zeros(shape, **options)
        self._grad_qdd_buffer = torch.zeros(shape, **options)
        # This is intentionally a metadata/cache allocation, not the CUDA
        # kernel's packed 20-float-per-link representation.
        self._forward_cache = torch.zeros(
            (total_batch, self._n_links, 6), **options
        )
        self._total_batch = total_batch

    def _ensure_buffer_capacity(self, total_batch: int) -> None:
        if self._tau_buffer is None or self._tau_buffer.shape[0] < total_batch:
            self._allocate_buffers(total_batch)

    def _sync_gravity(self) -> None:
        """Apply public config mutations before the next eager RNEA call."""
        gravity = self.config.get_gravity()
        with torch.no_grad():
            self._model.gravity.copy_(gravity)
        self._gravity_spatial = self.config.get_gravity_spatial()

    def _validate_state_channels(
        self, state: JointState, *, require_acceleration: bool
    ) -> torch.Size:
        """Enforce the tensor contract before flattening batch/horizon ranks."""
        if state.velocity is None:
            raise ValueError("joint velocity is required")
        if require_acceleration and state.acceleration is None:
            raise ValueError("joint acceleration is required")
        shape = state.position.shape
        if state.position.ndim not in (1, 2, 3) or shape[-1] != self._model.dof:
            raise ValueError(f"joint state must end in {self._model.dof} values")
        channels = (state.position, state.velocity, state.acceleration)
        for channel in channels:
            if channel is None:
                continue
            if channel.shape != shape:
                raise ValueError("position, velocity, and acceleration must have identical shapes")
            if (
                channel.device.type != self._model.device.type
                or channel.dtype != self._model.dtype
            ):
                raise TypeError("joint state must match the dynamics device and dtype")
            if not bool(torch.isfinite(channel).all().item()):
                raise ValueError("joint state must contain only finite values")
        return shape

    def _check_and_reorder_joints(self, joint_state: JointState) -> JointState:
        if joint_state.joint_names is None:
            return joint_state
        source = tuple(joint_state.joint_names)
        if len(source) != self._model.dof or set(source) != set(self._model.joint_names):
            raise ValueError("joint names do not match dynamics model")
        if len(set(source)) != len(source):
            raise ValueError("joint names must be unique")
        if source == self._backend_joint_names:
            return joint_state
        if (
            self._joint_reorder_source != source
            or self._joint_reorder_indices is None
            or self._joint_reorder_indices.device != joint_state.position.device
        ):
            self._joint_reorder_source = source
            self._joint_reorder_indices = torch.tensor(
                [source.index(name) for name in self._backend_joint_names],
                device=joint_state.position.device,
                dtype=torch.long,
            )
        assert self._joint_reorder_indices is not None
        order = self._joint_reorder_indices
        return JointState(
            position=joint_state.position.index_select(-1, order),
            velocity=None if joint_state.velocity is None else joint_state.velocity.index_select(-1, order),
            acceleration=None if joint_state.acceleration is None else joint_state.acceleration.index_select(-1, order),
            jerk=None if joint_state.jerk is None else joint_state.jerk.index_select(-1, order),
            joint_names=list(self._model.joint_names),
        )

    def _flatten_external_wrenches(
        self, f_ext: torch.Tensor, state_shape: torch.Size
    ) -> torch.Tensor:
        """Validate/broadcast local ``[moment, force]`` wrenches by link."""
        if not isinstance(f_ext, torch.Tensor):
            raise TypeError("f_ext must be a torch.Tensor")
        # ``mps`` and ``mps:0`` are the same execution device but are not
        # equal ``torch.device`` values on every PyTorch release.
        if f_ext.device.type != self._model.device.type or f_ext.dtype != self._model.dtype:
            raise TypeError("f_ext must match the dynamics device and dtype")
        if f_ext.shape[-2:] != (self._n_links, 6):
            raise ValueError(
                f"f_ext must end in [{self._n_links}, 6], got {tuple(f_ext.shape)}"
            )
        if not bool(torch.isfinite(f_ext).all().item()):
            raise ValueError("f_ext must contain only finite values")
        total_batch = int(np.prod(state_shape[:-1])) if len(state_shape) > 1 else 1
        if f_ext.ndim == 2:
            return f_ext.unsqueeze(0).expand(total_batch, -1, -1)
        if tuple(f_ext.shape[:-2]) == tuple(state_shape[:-1]):
            return f_ext.reshape(total_batch, self._n_links, 6)
        if f_ext.ndim == 3 and f_ext.shape[0] == total_batch:
            return f_ext
        raise ValueError(
            "f_ext leading dimensions must match joint_state.position, "
            "or use [num_links, 6] for broadcast"
        )

    def _external_wrench_torque(self, q: torch.Tensor, f_ext: torch.Tensor) -> torch.Tensor:
        """Map link-frame external wrenches to generalized effort.

        RNEA defines external wrenches as loads subtracted from internal
        wrench.  The composed backend computes that same contribution as
        ``J^T wrench`` after rotating each local wrench into the world frame.
        """
        fk = tree_forward_kinematics(self._model, q)
        rotation = fk.transforms[:, :, :3, :3]
        moment_world = (rotation @ f_ext[..., :3, None]).squeeze(-1)
        force_world = (rotation @ f_ext[..., 3:, None]).squeeze(-1)
        jacobian = fk.geometric_jacobian
        return (
            torch.einsum("blid,bli->bd", jacobian[:, :, :3], force_world)
            + torch.einsum("blid,bli->bd", jacobian[:, :, 3:], moment_world)
        )

    def compute_inverse_dynamics(
        self, joint_state: JointState, f_ext: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        state = self._check_and_reorder_joints(joint_state)
        shape = self._validate_state_channels(state, require_acceleration=True)
        self._sync_gravity()
        total_batch = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        self._ensure_buffer_capacity(total_batch)
        q = state.position.reshape(total_batch, shape[-1])
        qd = state.velocity.reshape(total_batch, shape[-1])
        qdd = state.acceleration.reshape(total_batch, shape[-1])
        result = self._model.inverse_dynamics(WholeBodyState(
            q, qd, qdd,
            tuple(self._model.joint_names),
        )).torque
        if f_ext is not None:
            flat_wrenches = self._flatten_external_wrenches(f_ext, shape)
            if flat_wrenches.requires_grad:
                if (
                    self._grad_f_ext_buffer is None
                    or self._grad_f_ext_buffer.shape[0] < total_batch
                ):
                    self._grad_f_ext_buffer = torch.zeros(
                        (total_batch, self._n_links, 6),
                        device=self._model.device,
                        dtype=self._model.dtype,
                    )
            result = result - self._external_wrench_torque(q, flat_wrenches)
        return result.reshape(shape)

    def compute_forward_dynamics(
        self, joint_state: JointState, torque: torch.Tensor
    ) -> torch.Tensor:
        """Solve generalized accelerations on CPU/MPS for portable consumers."""
        state = self._check_and_reorder_joints(joint_state)
        shape = self._validate_state_channels(state, require_acceleration=False)
        if torque.shape != shape:
            raise ValueError("joint state and torque must have matching [.., dof] shapes")
        if torque.device.type != self._model.device.type or torque.dtype != self._model.dtype:
            raise TypeError("torque must match the dynamics device and dtype")
        if not bool(torch.isfinite(torque).all().item()):
            raise ValueError("torque must contain only finite values")
        self._sync_gravity()
        total_batch = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        result = forward_dynamics(
            self._model,
            state.position.reshape(total_batch, self._n_dof),
            state.velocity.reshape(total_batch, self._n_dof),
            torque.reshape(total_batch, self._n_dof),
        ).acceleration
        return result.reshape(shape)

    def get_mass_matrix(self, joint_position: torch.Tensor) -> torch.Tensor:
        """Return a differentiable mass matrix for direct portable adapters."""
        if joint_position.ndim not in (1, 2):
            raise ValueError("joint_position must have shape [dof] or [batch, dof]")
        if joint_position.shape[-1] != self._n_dof:
            raise ValueError(f"joint_position must end in {self._n_dof} values")
        if (
            joint_position.device.type != self._model.device.type
            or joint_position.dtype != self._model.dtype
        ):
            raise TypeError("joint_position must match the dynamics device and dtype")
        if not bool(torch.isfinite(joint_position).all().item()):
            raise ValueError("joint_position must contain only finite values")
        self._sync_gravity()
        return mass_matrix(self._model, joint_position)

    def rollout(
        self, initial_state: JointState, torque: torch.Tensor, timestep: float | torch.Tensor
    ):
        """Semi-implicit Euler rollout; CUDA rollout kernels are not emulated."""
        state = self._check_and_reorder_joints(initial_state)
        self._validate_state_channels(state, require_acceleration=False)
        self._sync_gravity()
        return rollout_dynamics(
            self._model,
            WholeBodyState(state.position, state.velocity, state.acceleration, tuple(self._model.joint_names)),
            torque,
            timestep,
        )

    def _get_link_index(self, link_name: str) -> int:
        try:
            return self._model.link_names.index(link_name)
        except ValueError as error:
            raise ValueError(f"unknown link: {link_name}") from error

    def update_link_mass(self, link_name: str, mass: float) -> None:
        if not np.isfinite(mass) or mass < 0:
            raise ValueError("mass must be nonnegative")
        index = self._get_link_index(link_name)
        self._model.mass[index] = mass
        self.kinematics_config.update_link_mass(link_name, mass)
        self._link_masses_com = self.kinematics_config.link_masses_com

    def update_link_com(self, link_name: str, com: torch.Tensor) -> None:
        value = torch.as_tensor(com, device=self._model.device, dtype=self._model.dtype)
        if value.shape != (3,):
            raise ValueError("com must have shape [3]")
        self._model.com[self._get_link_index(link_name)].copy_(value)
        self.kinematics_config.update_link_com(link_name, value)
        self._link_masses_com = self.kinematics_config.link_masses_com

    def update_link_inertia(self, link_name: str, inertia: torch.Tensor) -> None:
        value = torch.as_tensor(inertia, device=self._model.device, dtype=self._model.dtype)
        if value.shape not in {(3, 3), (6,)}:
            raise ValueError("inertia must have shape [3,3] or [6]")
        if value.shape == (6,):
            xx, yy, zz, xy, xz, yz = value
            value = torch.stack((
                torch.stack((xx, xy, xz)), torch.stack((xy, yy, yz)),
                torch.stack((xz, yz, zz)),
            ))
        self._model.inertia[self._get_link_index(link_name)].copy_(value)
        self.kinematics_config.update_link_inertia(link_name, value)
        self._link_inertias = self.kinematics_config.link_inertias

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        if mass is None and com is None and inertia is None:
            raise ValueError("at least one inertial property must be provided")
        if mass is not None:
            self.update_link_mass(link_name, mass)
        if com is not None:
            self.update_link_com(link_name, com)
        if inertia is not None:
            self.update_link_inertia(link_name, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]]
    ) -> None:
        if not link_properties:
            raise ValueError("link_properties cannot be empty")
        for name, values in link_properties.items():
            if not values:
                raise ValueError(f"no inertial properties specified for {name}")
            unexpected = set(values) - {"mass", "com", "inertia"}
            if unexpected:
                raise ValueError(f"unknown inertial properties: {sorted(unexpected)}")
            self.update_link_inertial(name, **values)


__all__ = ["Dynamics", "_compute_threads_per_batch"]
