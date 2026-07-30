"""Differentiable portable rigid-body inverse dynamics."""

from __future__ import annotations

from typing import Optional, Union
from copy import deepcopy

import numpy as np
import torch

from curobo._src.state.state_joint import JointState
from curobo_metal.ops.whole_body import WholeBodyState

from .dynamics_cfg import DynamicsCfg


class Dynamics:
    def __init__(self, config: DynamicsCfg) -> None:
        self.config = config
        self.kinematics_config = config.kinematics_config
        robot_cfg = deepcopy(self.kinematics_config.robot_cfg)
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
        self._model.gravity.copy_(config.device_cfg.to_device(config.gravity))
        self.batch_size = 1
        self.horizon = 1

    def setup_batch_size(self, batch_size: int, horizon: int = 1) -> None:
        if batch_size < 1 or horizon < 1:
            raise ValueError("batch_size and horizon must be positive")
        self.batch_size, self.horizon = batch_size, horizon

    def _check_and_reorder_joints(self, joint_state: JointState) -> JointState:
        if joint_state.joint_names is None:
            return joint_state
        if set(joint_state.joint_names) != set(self._model.joint_names):
            raise ValueError("joint names do not match dynamics model")
        order = [joint_state.joint_names.index(name) for name in self._model.joint_names]
        return JointState(
            position=joint_state.position[..., order],
            velocity=None if joint_state.velocity is None else joint_state.velocity[..., order],
            acceleration=None if joint_state.acceleration is None else joint_state.acceleration[..., order],
            jerk=None if joint_state.jerk is None else joint_state.jerk[..., order],
            joint_names=list(self._model.joint_names),
        )

    def compute_inverse_dynamics(
        self, joint_state: JointState, f_ext: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if f_ext is not None and bool(torch.count_nonzero(f_ext).item()):
            raise NotImplementedError("nonzero external spatial forces are not yet supported")
        state = self._check_and_reorder_joints(joint_state)
        if state.velocity is None or state.acceleration is None:
            raise ValueError("joint velocity and acceleration are required")
        shape = state.position.shape
        if shape[-1] != self._model.dof:
            raise ValueError(f"joint state must end in {self._model.dof} values")
        result = self._model.inverse_dynamics(WholeBodyState(
            state.position.reshape(-1, shape[-1]),
            state.velocity.reshape(-1, shape[-1]),
            state.acceleration.reshape(-1, shape[-1]),
            tuple(self._model.joint_names),
        )).torque
        return result.reshape(shape)

    def _get_link_index(self, link_name: str) -> int:
        try:
            return self._model.link_names.index(link_name)
        except ValueError as error:
            raise ValueError(f"unknown link: {link_name}") from error

    def update_link_mass(self, link_name: str, mass: float) -> None:
        if mass < 0:
            raise ValueError("mass must be nonnegative")
        self._model.mass[self._get_link_index(link_name)] = mass

    def update_link_com(self, link_name: str, com: torch.Tensor) -> None:
        value = torch.as_tensor(com, device=self._model.device, dtype=self._model.dtype)
        if value.shape != (3,):
            raise ValueError("com must have shape [3]")
        self._model.com[self._get_link_index(link_name)].copy_(value)

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

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        if mass is not None:
            self.update_link_mass(link_name, mass)
        if com is not None:
            self.update_link_com(link_name, com)
        if inertia is not None:
            self.update_link_inertia(link_name, inertia)

    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]]
    ) -> None:
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)


__all__ = ["Dynamics"]
