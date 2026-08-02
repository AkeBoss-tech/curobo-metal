"""Portable URDF robot parser with cuRoboV2-compatible records."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import xml.etree.ElementTree as ET

import numpy as np

from curobo._src.geom.types import Mesh, Obstacle, Sphere
from curobo._src.robot.types import JointType, LinkParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo_metal.config.loaders import load_urdf

from .parser_base import RobotParser


def _rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _joint_type(kind: str, axis: tuple[float, float, float]) -> JointType:
    if kind == "fixed":
        return JointType.FIXED
    index = int(np.argmax(np.abs(axis)))
    sign = float(axis[index])
    if not np.allclose(np.abs(axis), np.eye(3)[index], atol=1e-6):
        raise NotImplementedError("cuRobo JointType supports only axis-aligned URDF joints")
    base = 0 if kind == "prismatic" else 3
    return JointType(base + index + (6 if sign < 0 else 0))


class UrdfRobotParser(RobotParser):
    def __init__(
        self,
        urdf_path: str,
        load_meshes: bool = False,
        mesh_root: str = "",
        extra_links: Optional[Dict[str, LinkParams]] = None,
        build_scene_graph: bool = False,
    ) -> None:
        if build_scene_graph:
            raise NotImplementedError("URDF scene-graph construction requires yourdfpy/trimesh")
        super().__init__(extra_links)
        self.urdf_path = str(Path(urdf_path).resolve())
        self.mesh_root = mesh_root
        self.load_meshes = load_meshes
        self._robot = load_urdf(self.urdf_path)
        self._root = ET.parse(self.urdf_path).getroot()
        self._joint_by_child = {joint.child: joint for joint in self._robot.joints}
        self.build_link_parent()

    def _file_name_handler(self, fname: str) -> str:
        value = fname.removeprefix("package://")
        base = Path(self.mesh_root) if self.mesh_root else Path(self.urdf_path).parent
        return str((base / value).resolve())

    def build_link_parent(self) -> None:
        self.link_parent = {joint.child: joint.parent for joint in self._robot.joints}
        for name, params in self.extra_links.items():
            if params.parent_link_name is not None:
                self.link_parent[name] = params.parent_link_name

    def _get_joint_name(self, idx: int) -> str:
        return self._robot.joints[idx].name

    def _get_joint_limits(self, joint: object) -> Tuple[Dict[str, float], str]:
        return {
            "lower": float(joint.limits.lower), "upper": float(joint.limits.upper),
            "velocity": float(joint.limits.velocity), "effort": float(joint.limits.effort),
        }, joint.name

    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams:
        if link_name in self.extra_links:
            return self.extra_links[link_name]
        links = {link.name: link for link in self._robot.links}
        if link_name not in links:
            raise ValueError(f"unknown robot link: {link_name}")
        link = links[link_name]
        joint = self._joint_by_child.get(link_name)
        if base or joint is None:
            return LinkParams(
                link_name, "base_joint", JointType.FIXED,
                np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1),
                link_mass=link.mass, link_com=np.asarray(link.com),
                link_inertia=np.asarray(link.inertia),
            )
        transform = np.concatenate(
            [_rotation(joint.rpy), np.asarray(joint.xyz).reshape(3, 1)], axis=1
        )
        return LinkParams(
            link_name=link_name, joint_name=joint.name,
            joint_type=_joint_type(joint.kind, joint.axis),
            fixed_transform=transform, parent_link_name=joint.parent,
            child_link_name=joint.child,
            joint_limits=[joint.limits.lower, joint.limits.upper],
            joint_axis=np.asarray(joint.axis),
            joint_velocity_limits=[-joint.limits.velocity, joint.limits.velocity],
            joint_offset=[joint.mimic_multiplier, joint.mimic_offset],
            mimic_joint_name=joint.mimic_joint,
            joint_effort_limit=[joint.limits.effort],
            link_mass=link.mass, link_com=np.asarray(link.com),
            link_inertia=np.asarray(link.inertia),
        )

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = "") -> None:
        if mesh_dir:
            self.mesh_root = mesh_dir

    def get_urdf_string(self) -> str:
        return Path(self.urdf_path).read_text(encoding="utf-8")

    def _link_element(self, link_name: str) -> ET.Element:
        result = next((x for x in self._root.findall("link") if x.get("name") == link_name), None)
        if result is None:
            raise ValueError(f"unknown robot link: {link_name}")
        return result

    def get_link_geometry(
        self, link_name: str, use_collision_mesh: bool = False
    ) -> List[Obstacle]:
        tag = "collision" if use_collision_mesh else "visual"
        result: List[Obstacle] = []
        for index, node in enumerate(self._link_element(link_name).findall(tag)):
            origin = node.find("origin")
            xyz = [0.0, 0.0, 0.0] if origin is None else [
                float(v) for v in origin.get("xyz", "0 0 0").split()
            ]
            geometry = node.find("geometry")
            if geometry is None:
                continue
            sphere = geometry.find("sphere")
            mesh = geometry.find("mesh")
            if sphere is not None:
                result.append(Sphere(
                    name=f"{link_name}_{tag}_{index}", position=xyz,
                    radius=float(sphere.get("radius", "0")),
                ))
            elif mesh is not None:
                raise NotImplementedError(
                    "file-backed URDF mesh loading requires the optional trimesh backend"
                )
            else:
                result.append(Obstacle(
                    name=f"{link_name}_{tag}_{index}",
                    pose=xyz + [1.0, 0.0, 0.0, 0.0],
                ))
        return result

    def get_link_mesh(
        self, link_name: str, use_collision_mesh: bool = False
    ) -> Union[Mesh, None]:
        del link_name, use_collision_mesh
        raise NotImplementedError(
            "file-backed URDF mesh loading requires the optional trimesh backend"
        )

    @property
    def root_link(self) -> str:
        return self._robot.base_link

    def get_link_names_from_urdf(self) -> List[str]:
        return [x.name for x in self._robot.links]

    def get_joint_names_from_urdf(self) -> List[str]:
        return [x.name for x in self._robot.joints]

    def get_link_names(self) -> List[str]:
        return self.get_link_names_from_urdf() + list(self.extra_links)


CuroboMesh = Mesh

__all__ = ["UrdfRobotParser"]
