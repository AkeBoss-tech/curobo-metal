"""Mutable RobotCfg-style model and production configuration adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from curobo_metal.reference.forward_kinematics import SerialRobot
from curobo_metal.reference.tree_kinematics import TreeRobot
from curobo_metal.types import DeviceCfg


class UnsupportedConfigError(NotImplementedError):
    """A configuration requires a deliberately unavailable runtime."""


@dataclass
class JointLimits:
    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = np.inf
    effort: float = np.inf


@dataclass
class JointConfig:
    name: str
    kind: str
    parent: str
    child: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits: JointLimits = field(default_factory=JointLimits)
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class LinkConfig:
    name: str
    mass: float = 0.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )


@dataclass
class CollisionSphere:
    link_name: str
    center: tuple[float, float, float]
    radius: float


@dataclass
class CSpaceConfig:
    joint_names: list[str] = field(default_factory=list)
    default_joint_position: list[float] = field(default_factory=list)
    max_velocity: float | list[float] | None = None
    max_acceleration: float | list[float] | None = None
    max_jerk: float | list[float] | None = None
    cspace_distance_weight: list[float] | None = None
    null_space_weight: list[float] | None = None


@dataclass
class RobotCfg:
    """Mutable, serialization-friendly robot model.

    ``kinematics`` returns ``self`` so common ``robot_cfg.kinematics.cspace``
    call sites remain adapter-compatible.
    """

    name: str
    base_link: str
    tool_frames: list[str]
    links: list[LinkConfig]
    joints: list[JointConfig]
    collision_spheres: list[CollisionSphere] = field(default_factory=list)
    cspace: CSpaceConfig = field(default_factory=CSpaceConfig)
    collision_link_names: list[str] = field(default_factory=list)
    self_collision_ignore: dict[str, list[str]] = field(default_factory=dict)
    self_collision_buffer: dict[str, float] = field(default_factory=dict)
    urdf_path: str | None = None
    xrdf_path: str | None = None
    source_path: str | None = None
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    metadata: dict[str, Any] = field(default_factory=dict)
    dynamics: object | None = None

    @property
    def kinematics(self) -> "RobotCfg":
        return self

    @property
    def joint_names(self) -> list[str]:
        if self.cspace.joint_names:
            return self.cspace.joint_names
        return [
            joint.name for joint in self.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]

    @property
    def retract_config(self) -> torch.Tensor:
        values = self.cspace.default_joint_position or [0.0] * len(self.joint_names)
        return self.device_cfg.to_device(values)

    @classmethod
    def create(
        cls,
        data: Mapping[str, Any] | "RobotCfg" | str | Path,
        device_cfg: DeviceCfg = DeviceCfg(),
        load_collision_spheres: bool = True,
        num_envs: int = 1,
    ) -> "RobotCfg":
        del num_envs
        if isinstance(data, cls):
            data.device_cfg = device_cfg
            if not load_collision_spheres:
                data.collision_spheres = []
            return data
        from .loaders import load_robot_config
        result = load_robot_config(data, device_cfg=device_cfg)
        if not load_collision_spheres:
            result.collision_spheres = []
        return result

    @classmethod
    def from_basic(
        cls,
        urdf_path: str | Path,
        base_link: str,
        tool_frames: Sequence[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_dynamics: bool = False,
    ) -> "RobotCfg":
        from .loaders import load_urdf
        result = load_urdf(urdf_path, base_link=base_link, tool_frames=tool_frames)
        result.device_cfg = device_cfg
        if load_dynamics:
            result.dynamics = result.to_whole_body_model()
        return result

    def to_mapping(self) -> dict[str, Any]:
        spheres: dict[str, list[dict[str, Any]]] = {}
        for sphere in self.collision_spheres:
            spheres.setdefault(sphere.link_name, []).append(
                {"center": list(sphere.center), "radius": sphere.radius}
            )
        return {
            "robot_cfg": {
                "kinematics": {
                    "name": self.name,
                    "base_link": self.base_link,
                    "tool_frames": list(self.tool_frames),
                    "urdf_path": self.urdf_path,
                    "xrdf_path": self.xrdf_path,
                    "cspace": {
                        key: value for key, value in vars(self.cspace).items()
                        if value not in (None, [])
                    },
                    "collision_link_names": list(self.collision_link_names),
                    "collision_spheres": spheres,
                    "self_collision_ignore": self.self_collision_ignore,
                    "self_collision_buffer": self.self_collision_buffer,
                }
            }
        }

    def write_config(self, file_path: str | Path) -> None:
        from .loaders import dump_yaml
        Path(file_path).write_text(dump_yaml(self.to_mapping()), encoding="utf-8")

    def _tree_mapping(self) -> dict[str, Any]:
        link_by_name = {link.name: link for link in self.links}
        joint_by_child = {joint.child: joint for joint in self.joints}
        ordered_names = _topological_links(self.base_link, self.links, self.joints)
        index = {name: position for position, name in enumerate(ordered_names)}
        rows: list[dict[str, Any]] = []
        for name in ordered_names:
            link = link_by_name[name]
            joint = joint_by_child.get(name)
            joint_value: dict[str, Any] = {"type": "fixed"}
            parent = -1
            if joint is not None:
                parent = index[joint.parent]
                joint_value = {
                    "name": joint.name, "type": joint.kind, "axis": list(joint.axis),
                    "origin": {"xyz": list(joint.xyz), "rpy": list(joint.rpy)},
                }
                if joint.mimic_joint is not None:
                    joint_value["mimic"] = {
                        "joint": joint.mimic_joint,
                        "multiplier": joint.mimic_multiplier,
                        "offset": joint.mimic_offset,
                    }
            rows.append({
                "name": name, "parent": parent, "joint": joint_value,
                "inertial": {
                    "mass": link.mass, "com": list(link.com), "inertia": list(link.inertia)
                },
            })
        effort = {
            joint.name: joint.limits.effort for joint in self.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        }
        return {
            "name": self.name,
            "links": rows,
            "end_effectors": list(self.tool_frames),
            "effort_limits": [effort.get(name, np.inf) for name in self.joint_names],
        }

    def to_tree_robot(self) -> TreeRobot:
        return TreeRobot.from_dict(self._tree_mapping())

    def to_serial_robot(self) -> SerialRobot:
        if len(self.tool_frames) != 1:
            raise UnsupportedConfigError(
                "SerialRobot conversion requires exactly one tool frame; "
                "use to_tree_robot() for multi-effector robots"
            )
        joint_by_child = {joint.child: joint for joint in self.joints}
        path: list[JointConfig] = []
        child = self.tool_frames[0]
        while child != self.base_link:
            joint = joint_by_child.get(child)
            if joint is None:
                raise ValueError(
                    f"tool frame {self.tool_frames[0]!r} is not a descendant "
                    f"of base_link {self.base_link!r}"
                )
            if joint.mimic_joint is not None:
                raise UnsupportedConfigError(
                    "SerialRobot cannot preserve mimic joints; use to_tree_robot()"
                )
            path.append(joint)
            child = joint.parent
        path.reverse()
        mapping = {"name": self.name, "joints": []}
        for joint in path:
            mapping["joints"].append({
                "name": joint.child,
                "type": joint.kind,
                "axis": list(joint.axis),
                "origin": {"xyz": list(joint.xyz), "rpy": list(joint.rpy)},
            })
        return SerialRobot.from_dict(mapping)

    def to_kinematic_chain(
        self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Any:
        from curobo_metal.ops.kinematics import KinematicChain
        return KinematicChain(
            self.to_serial_robot(),
            device=self.device_cfg.device if device is None else device,
            dtype=self.device_cfg.dtype if dtype is None else dtype,
        )

    def to_whole_body_model(
        self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Any:
        from curobo_metal.ops.whole_body import WholeBodyModel
        return WholeBodyModel(
            self.to_tree_robot(),
            device=self.device_cfg.device if device is None else device,
            dtype=self.device_cfg.dtype if dtype is None else dtype,
        )

    def to_collision_inputs(
        self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.device_cfg.device if device is None else torch.device(device)
        actual_dtype = self.device_cfg.dtype if dtype is None else dtype
        names = _topological_links(self.base_link, self.links, self.joints)
        indices = {name: index for index, name in enumerate(names[1:])}
        missing = sorted({sphere.link_name for sphere in self.collision_spheres} - set(indices))
        if missing:
            raise ValueError(f"collision spheres reference unavailable/root links: {missing}")
        values = [[*sphere.center, sphere.radius] for sphere in self.collision_spheres]
        links = [indices[sphere.link_name] for sphere in self.collision_spheres]
        return (
            torch.tensor(values, device=target, dtype=actual_dtype).reshape(-1, 4),
            torch.tensor(links, device=target, dtype=torch.int64),
        )


def _topological_links(
    base_link: str, links: Sequence[LinkConfig], joints: Sequence[JointConfig]
) -> list[str]:
    link_names = [link.name for link in links]
    if base_link not in link_names:
        raise ValueError(f"base_link {base_link!r} is absent from URDF links")
    children: dict[str, list[str]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint.child)
    ordered: list[str] = []
    queue = [base_link]
    while queue:
        name = queue.pop(0)
        if name in ordered:
            raise ValueError("joint graph contains a cycle or duplicate child")
        ordered.append(name)
        queue.extend(children.get(name, []))
    if set(ordered) != set(link_names):
        missing = sorted(set(link_names) - set(ordered))
        raise ValueError(f"URDF contains links disconnected from base_link: {missing}")
    return ordered


def _matrix_origin(matrix: np.ndarray) -> dict[str, list[float]]:
    from curobo_metal.reference.forward_kinematics import _rpy_matrix
    rotation = matrix[:3, :3]
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1, 1)))
    if abs(np.cos(pitch)) > 1e-9:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    rpy = [roll, pitch, yaw]
    if not np.allclose(_rpy_matrix(rpy), rotation, atol=1e-8):
        raise ValueError("could not recover URDF RPY from transform")
    return {"xyz": matrix[:3, 3].tolist(), "rpy": rpy}
