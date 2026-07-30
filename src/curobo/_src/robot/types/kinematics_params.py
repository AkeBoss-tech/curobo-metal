"""Portable representation of the pinned kinematics parameter object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from curobo._src.state.state_joint import JointState


@dataclass
class KinematicsParams:
    """Tensor/model metadata consumed by :class:`curobo.kinematics.Kinematics`.

    The CUDA implementation exposes a very large tensor record.  This portable
    implementation preserves its commonly consumed public attributes while the
    production tree-kinematics backend owns the compiled representation.
    """

    robot_cfg: Any

    @property
    def num_dof(self) -> int:
        return len(self.robot_cfg.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot_cfg.joint_names)

    @property
    def non_fixed_joint_names(self) -> list[str]:
        return [
            joint.name for joint in self.robot_cfg.joints if joint.kind != "fixed"
        ]

    @property
    def base_link(self) -> str:
        return self.robot_cfg.base_link

    @property
    def cspace(self) -> Any:
        return self.robot_cfg.cspace

    @property
    def total_spheres(self) -> int:
        return len(self.robot_cfg.collision_spheres)

    @property
    def link_spheres(self) -> torch.Tensor:
        spheres, _ = self.robot_cfg.to_collision_inputs()
        return spheres.unsqueeze(0)

    @property
    def mesh_link_names(self) -> list[str]:
        return list(self.robot_cfg.metadata.get("mesh_link_names", []))

    @property
    def lock_jointstate(self) -> JointState:
        locked = self.robot_cfg.metadata.get("lock_joints", {})
        return JointState.from_position(
            self.robot_cfg.device_cfg.to_device(list(locked.values())),
            joint_names=list(locked),
        )

    def make_contiguous(self) -> None:
        """Compatibility no-op: tensors are created contiguous by the backend."""

    def copy_(self, other: "KinematicsParams") -> "KinematicsParams":
        if not isinstance(other, KinematicsParams):
            raise TypeError("other must be KinematicsParams")
        self.robot_cfg = other.robot_cfg
        return self

