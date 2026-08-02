"""Portable representation of the pinned kinematics parameter object."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
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
    def device_cfg(self):
        """Device policy shared by every tensor exposed by this record."""
        return self.robot_cfg.device_cfg

    @property
    def cspace(self) -> Any:
        return self.robot_cfg.cspace

    @property
    def debug(self) -> Any:
        return self.robot_cfg.metadata.get("debug")

    @property
    def joint_limits(self):
        """Return portable tensor joint limits in the active-joint order.

        The config model stores scalar limits on the URDF joints; consumers of
        the historical ``KinematicsParams`` record expect the tensor-valued
        ``JointLimits`` view.  Constructing it here keeps mutations of the
        source robot visible and keeps CPU/MPS placement deterministic.
        """
        from .joint_limits import JointLimits

        names = self.joint_names
        by_name = {joint.name: joint for joint in self.robot_cfg.joints}
        joints = [by_name[name] for name in names]

        def bounds(values: list[tuple[float, float]]) -> torch.Tensor:
            return self.device_cfg.to_device(values).transpose(0, 1).contiguous()

        return JointLimits(
            names,
            bounds([(joint.limits.lower, joint.limits.upper) for joint in joints]),
            bounds([(-joint.limits.velocity, joint.limits.velocity) for joint in joints]),
            bounds([(-10.0, 10.0) for _ in joints]),
            bounds([(-500.0, 500.0) for _ in joints]),
            bounds([(-joint.limits.effort, joint.limits.effort) for joint in joints]),
            self.device_cfg,
        )

    @property
    def total_spheres(self) -> int:
        return len(self.robot_cfg.collision_spheres)

    @property
    def link_spheres(self) -> torch.Tensor:
        if self._link_spheres is None:
            self._link_spheres = self.robot_cfg.device_cfg.to_device([
                [*sphere.center, sphere.radius]
                for sphere in self.robot_cfg.collision_spheres
            ]).reshape(1, -1, 4)
            self.reference_link_spheres = self._link_spheres.clone()
        return self._link_spheres

    @property
    def link_sphere_idx_map(self) -> torch.Tensor:
        mapping = self.link_name_to_idx_map
        return torch.tensor(
            [mapping[sphere.link_name] for sphere in self.robot_cfg.collision_spheres],
            dtype=torch.int64,
            device=self.robot_cfg.device_cfg.device,
        )

    @property
    def link_name_to_idx_map(self) -> dict[str, int]:
        return {link.name: index for index, link in enumerate(self.robot_cfg.links)}

    @property
    def link_map(self) -> torch.Tensor:
        """Stable link index for each link, matching the tree model order."""
        return torch.arange(self.num_links, dtype=torch.int64, device=self.device_cfg.device)

    @property
    def joint_map(self) -> torch.Tensor:
        """Stable active-joint index map; fixed and mimic joints use ``-1``."""
        active = {name: index for index, name in enumerate(self.joint_names)}
        return torch.tensor(
            [active.get(joint.name, -1) for joint in self.robot_cfg.joints],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def joint_map_type(self) -> torch.Tensor:
        kinds = {"fixed": -1, "prismatic": 0, "revolute": 1}
        return torch.tensor(
            [kinds.get(joint.kind, -2) for joint in self.robot_cfg.joints],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def joint_offset_map(self) -> torch.Tensor:
        return self.device_cfg.to_device(
            [[joint.mimic_multiplier, joint.mimic_offset] for joint in self.robot_cfg.joints]
        ).reshape(-1, 2)

    @property
    def mimic_joints(self) -> dict[str, tuple[str, float, float]]:
        return {
            joint.name: (joint.mimic_joint, joint.mimic_multiplier, joint.mimic_offset)
            for joint in self.robot_cfg.joints
            if joint.mimic_joint is not None
        }

    @property
    def tool_frame_map(self) -> torch.Tensor:
        indices = self.link_name_to_idx_map
        return torch.tensor(
            [indices[name] for name in self.tool_frames],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def fixed_transforms(self) -> torch.Tensor:
        """[link, 4, 4] rest transforms, useful for portable introspection."""
        from curobo_metal.ops.whole_body import WholeBodyModel
        from curobo_metal.reference.tree_kinematics import TreeRobot

        mapping = self.robot_cfg._tree_mapping()
        # Rest FK does not consume inertia. Some widely-used URDF files carry
        # visualization inertias which are not PSD, whereas the reference tree
        # validates physical dynamics strictly.
        for link in mapping["links"]:
            link["inertial"]["inertia"] = [0.0] * 6
        model = WholeBodyModel(
            TreeRobot.from_dict(mapping),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        return model.origins.clone()

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
        result = KinematicsParams(deepcopy(self.robot_cfg))
        result._link_spheres = (
            None if self._link_spheres is None else self._link_spheres.clone()
        )
        result.reference_link_spheres = (
            None if self.reference_link_spheres is None
            else self.reference_link_spheres.clone()
        )
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
        indices = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, indices, 3] = -torch.abs(self.link_spheres[:, indices, 3])

    def enable_link_spheres(self, link_name: str) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, indices, 3] = torch.abs(self.link_spheres[:, indices, 3])

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
        parents = {joint.child: joint.parent for joint in self.robot_cfg.joints}
        max_depth = 0
        for link in self.all_link_names():
            depth, current = 0, link
            while current != self.base_link:
                current = parents.get(current)
                if current is None:
                    raise ValueError("robot tree contains a disconnected link")
                depth += 1
            max_depth = max(max_depth, depth)
        return max_depth + 1

    @property
    def max_level_width(self) -> int:
        parents = {joint.child: joint.parent for joint in self.robot_cfg.joints}
        widths: dict[int, int] = {}
        for link in self.all_link_names():
            depth, current = 0, link
            while current != self.base_link:
                current = parents[current]
                depth += 1
            widths[depth] = widths.get(depth, 0) + 1
        return max(widths.values(), default=0)

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
