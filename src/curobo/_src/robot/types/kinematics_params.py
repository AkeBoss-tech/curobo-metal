"""Portable representation of the pinned kinematics parameter object."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    reference_link_spheres: torch.Tensor | None = field(default=None, init=False)
    _link_spheres: torch.Tensor | None = field(default=None, init=False, repr=False)

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
        if self._link_spheres is None:
            spheres, _ = self.robot_cfg.to_collision_inputs()
            self._link_spheres = spheres.unsqueeze(0)
            self.reference_link_spheres = self._link_spheres.clone()
        return self._link_spheres

    @property
    def link_sphere_idx_map(self) -> torch.Tensor:
        _, indices = self.robot_cfg.to_collision_inputs()
        return indices

    @property
    def link_name_to_idx_map(self) -> dict[str, int]:
        return {link.name: index for index, link in enumerate(self.robot_cfg.links)}

    @property
    def tool_frames(self) -> list[str]:
        return list(self.robot_cfg.tool_frames)

    def all_link_names(self) -> list[str]:
        return [link.name for link in self.robot_cfg.links]

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
        if self._link_spheres is not None:
            self._link_spheres = self._link_spheres.contiguous()

    def copy_(self, other: "KinematicsParams") -> "KinematicsParams":
        if not isinstance(other, KinematicsParams):
            raise TypeError("other must be KinematicsParams")
        self.robot_cfg = other.robot_cfg
        self._link_spheres = (
            None if other._link_spheres is None else other._link_spheres.clone()
        )
        self.reference_link_spheres = (
            None if other.reference_link_spheres is None
            else other.reference_link_spheres.clone()
        )
        return self

    def clone(self) -> "KinematicsParams":
        result = KinematicsParams(self.robot_cfg)
        result.copy_(self)
        return result

    def validate_shapes(self) -> None:
        if self.num_dof != len(self.joint_names):
            raise ValueError("num_dof and joint_names disagree")
        if self.link_spheres.ndim != 3 or self.link_spheres.shape[-1] != 4:
            raise ValueError("link_spheres must have shape [env, sphere, 4]")

    def load_cspace_cfg_from_kinematics(self) -> None:
        if not self.robot_cfg.cspace.joint_names:
            self.robot_cfg.cspace.joint_names = self.joint_names

    def get_sphere_index_from_link_name(self, link_name: str) -> torch.Tensor:
        values = [
            index for index, sphere in enumerate(self.robot_cfg.collision_spheres)
            if sphere.link_name == link_name
        ]
        if not values:
            return torch.empty(0, dtype=torch.int64, device=self.robot_cfg.device_cfg.device)
        return torch.tensor(values, dtype=torch.int64, device=self.robot_cfg.device_cfg.device)

    def update_link_spheres(
        self,
        link_name: str,
        sphere_position_radius: torch.Tensor,
        start_sph_idx: int = 0,
        config_idx: int | None = None,
    ) -> None:
        env = 0 if config_idx is None else config_idx
        indices = self.get_sphere_index_from_link_name(link_name)
        values = torch.as_tensor(
            sphere_position_radius,
            device=self.link_spheres.device,
            dtype=self.link_spheres.dtype,
        ).reshape(-1, 4)
        target = indices[start_sph_idx:start_sph_idx + len(values)]
        if len(target) != len(values):
            raise ValueError("too many sphere values for link")
        self.link_spheres[env, target] = values

    def get_link_spheres(self, link_name: str, config_idx: int = 0) -> torch.Tensor:
        return self.link_spheres[config_idx, self.get_sphere_index_from_link_name(link_name)]

    def get_reference_link_spheres(self, link_name: str, config_idx: int = 0) -> torch.Tensor:
        self.link_spheres
        return self.reference_link_spheres[
            config_idx, self.get_sphere_index_from_link_name(link_name)
        ]

    def get_number_of_spheres(self, link_name: str) -> int:
        return int(self.get_sphere_index_from_link_name(link_name).numel())

    def disable_link_spheres(self, link_name: str) -> None:
        self.get_link_spheres(link_name)[..., 3] = -torch.abs(
            self.get_link_spheres(link_name)[..., 3]
        )

    def enable_link_spheres(self, link_name: str) -> None:
        self.get_link_spheres(link_name)[..., 3] = torch.abs(
            self.get_link_spheres(link_name)[..., 3]
        )

    def reset_link_spheres(self, link_name: str) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, indices] = self.reference_link_spheres[:, indices]

    def get_link_masses_com(self, link_name: str) -> torch.Tensor:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        return self.robot_cfg.device_cfg.to_device([*link.com, link.mass])

    def update_link_mass(self, link_name: str, mass: float) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        link.mass = float(mass)

    def update_link_com(self, link_name: str, com: torch.Tensor) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        values = torch.as_tensor(com).reshape(-1)
        if values.numel() != 3:
            raise ValueError("com must have three values")
        link.com = tuple(values.cpu().tolist())

    def update_link_inertia(self, link_name: str, inertia: torch.Tensor) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        values = torch.as_tensor(inertia).reshape(-1)
        if values.numel() not in (6, 9):
            raise ValueError("inertia must have six values or a 3x3 matrix")
        if values.numel() == 9:
            matrix = values.reshape(3, 3)
            values = matrix[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        link.inertia = tuple(values.cpu().tolist())

    def get_link_inertia(self, link_name: str) -> torch.Tensor:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        return self.robot_cfg.device_cfg.to_device(link.inertia)

    def get_robot_collision_geometry(self):
        from .collision_geometry import RobotCollisionGeometry
        return RobotCollisionGeometry(self.link_sphere_idx_map.clone(), self.num_links)

    @property
    def num_pose_links(self) -> int:
        return len(self.tool_frames)

    @property
    def num_links(self) -> int:
        return len(self.robot_cfg.links)

    @property
    def num_spheres(self) -> int:
        return self.total_spheres

    @property
    def num_envs(self) -> int:
        return self.link_spheres.shape[0]

    @property
    def n_tree_levels(self) -> int:
        return max(1, len(self.robot_cfg.links))

    def export_to_urdf(
        self,
        robot_name: str = "robot",
        output_path: str | None = None,
        include_spheres: bool = False,
        kinematics_parser=None,
    ) -> str:
        del robot_name, include_spheres
        parser = kinematics_parser
        if parser is None and self.robot_cfg.urdf_path:
            from curobo._src.robot.parser import UrdfRobotParser
            parser = UrdfRobotParser(self.robot_cfg.urdf_path)
        if parser is None:
            raise NotImplementedError("URDF export requires a source URDF parser")
        value = parser.get_urdf_string()
        if output_path is not None:
            from pathlib import Path
            Path(output_path).write_text(value, encoding="utf-8")
        return value
