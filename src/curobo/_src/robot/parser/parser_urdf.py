"""Portable URDF robot parser with cuRoboV2-compatible records."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union
import importlib
import math
import xml.etree.ElementTree as ET

import numpy as np
import torch

from curobo._src.geom.types import Cuboid, Cylinder, Mesh as CuroboMesh, Obstacle, Sphere
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import join_path
from curobo_metal.config.loaders import load_urdf

from .parser_base import RobotParser


def _optional_upstream_module(name: str):
    """Retain an inspectable upstream alias without making it a runtime dependency.

    The portable parser uses ``ElementTree`` and its own lightweight loader;
    importing the CUDA-side ``yourdfpy``/``lxml`` stack would make ordinary
    URDF parsing fail in a minimal Metal installation.  Publish ``None`` when
    either optional integration is absent, matching the existing explicit
    optional-boundary behavior elsewhere in the port.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


# These are public aliases in the pinned module.  They are deliberately lazy
# optional integrations: none of the portable parsing path depends on them.
yourdfpy = _optional_upstream_module("yourdfpy")
etree = _optional_upstream_module("lxml.etree")


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
    if not np.allclose(np.abs(axis), np.eye(3)[index], atol=1e-6):
        raise NotImplementedError("cuRobo JointType supports only axis-aligned URDF joints")
    base = 0 if kind == "prismatic" else 3
    # Pinned V2 canonicalizes a negative URDF axis into the joint multiplier
    # rather than selecting one of its legacy *_NEG enum values.  This keeps
    # joint axes positive for the tree backend and preserves mimic semantics.
    return JointType(base + index)


def _pose_from_origin(node: Optional[ET.Element]) -> list[float]:
    """Convert a URDF origin to cuRobo's ``xyz + wxyz`` pose layout."""
    if node is None:
        xyz, rotation = np.zeros(3), np.eye(3)
    else:
        xyz = np.asarray([float(v) for v in node.get("xyz", "0 0 0").split()], dtype=float)
        rpy = tuple(float(v) for v in node.get("rpy", "0 0 0").split())
        rotation = _rotation(rpy)
    # Stable branch-free matrix-to-quaternion conversion is already exposed
    # by the portable Pose implementation.  Materialize only this static
    # parser metadata on CPU; no accelerator work is involved.
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, xyz
    return Pose.from_matrix(matrix).to_list()


class UrdfRobotParser(RobotParser):
    def __init__(
        self,
        urdf_path,
        load_meshes: bool = False,
        mesh_root: str = "",
        extra_links: Optional[Dict[str, LinkParams]] = None,
        build_scene_graph: bool = False,
    ) -> None:
        del build_scene_graph
        super().__init__(extra_links)
        self.urdf_path = str(Path(urdf_path).resolve())
        self.mesh_root = mesh_root
        self._mesh_root = mesh_root
        self.load_meshes = load_meshes
        self._root = ET.parse(self.urdf_path).getroot()
        supported = {"fixed", "continuous", "revolute", "prismatic"}
        unsupported = [
            node.get("type") for node in self._root.findall("joint")
            if node.get("type") not in supported
        ]
        if unsupported:
            raise ValueError(f"unsupported URDF joint type: {unsupported[0]!r}")
        self._robot = load_urdf(self.urdf_path)
        self._robot.joint_map = {joint.name: joint for joint in self._robot.joints}
        self._robot.link_map = {
            str(link.get("name")): SimpleNamespace(
                visuals=[
                    SimpleNamespace(geometry=SimpleNamespace(mesh=(
                        None if visual.find("geometry/mesh") is None else SimpleNamespace(
                            filename=str(visual.find("geometry/mesh").get("filename"))
                        )
                    )))
                    for visual in link.findall("visual")
                ]
            )
            for link in self._root.findall("link")
        }
        self._joint_by_child = {joint.child: joint for joint in self._robot.joints}
        self._joint_element_by_name = {
            str(element.get("name")): element for element in self._root.findall("joint")
            if element.get("name")
        }
        self.build_link_parent()

    def _file_name_handler(self, fname: str) -> str:
        value = fname.removeprefix("package://")
        base = Path(self._mesh_root) if self._mesh_root else Path(self.urdf_path).parent
        return str((base / value).resolve())

    def build_link_parent(self):
        self.link_parent = {joint.child: joint.parent for joint in self._robot.joints}
        self._parent_map = {
            joint.child: {"parent": joint.parent, "jid": index, "joint_name": joint.name}
            for index, joint in enumerate(self._robot.joints)
        }
        for name, params in self.extra_links.items():
            if params.parent_link_name is not None:
                self.link_parent[name] = params.parent_link_name
                self._parent_map[name] = {
                    "parent": params.parent_link_name,
                    "joint_name": params.joint_name,
                }
            if params.child_link_name is not None:
                self.link_parent[params.child_link_name] = name
                self._parent_map[params.child_link_name] = {"parent": name}

    def _get_joint_name(self, idx: int) -> str:
        return self._robot.joints[idx].name

    def _get_joint_limits(self, joint: object) -> Tuple[Dict[str, float], str]:
        """Return V2-style finite limits and normalize continuous joints.

        ``load_urdf`` deliberately represents a continuous joint as a regular
        revolute one for solver compilation, so consult the original XML to
        retain the parser contract of a ``[-2π, 2π]`` window.  URDF permits
        omitted velocity/effort attributes; V2 uses a finite permissive
        default instead of leaking infinities into optimizer buffers.
        """
        element = self._joint_element_by_name.get(str(joint.name))
        raw_kind = None if element is None else element.get("type")
        kind = "revolute" if raw_kind == "continuous" else str(joint.kind)
        limits = joint.limits
        velocity = float(limits.velocity)
        effort = float(limits.effort)
        if not np.isfinite(velocity):
            velocity = 100.0
        if not np.isfinite(effort):
            effort = 100.0
        if raw_kind == "continuous":
            lower, upper = -2.0 * math.pi, 2.0 * math.pi
        else:
            lower, upper = float(limits.lower), float(limits.upper)
        return {"lower": lower, "upper": upper, "velocity": velocity, "effort": effort}, kind

    def get_link_parameters(self, link_name: str, base=False) -> LinkParams:
        extra = self._get_from_extra_links(link_name)
        if extra is not None:
            return extra
        links = {link.name: link for link in self._robot.links}
        if link_name not in links:
            raise ValueError(f"unknown robot link: {link_name}")
        link = links[link_name]
        joint = self._joint_by_child.get(link_name)
        # Use V2's robust small-mass/inertia defaults for incomplete URDF
        # inertial tags.  This avoids a singular inertial tree while retaining
        # authored nonzero values exactly.
        mass = float(link.mass) if float(link.mass) > 0.0 else 0.01
        com = np.asarray(link.com, dtype=float)
        inertia = np.asarray(link.inertia, dtype=float)
        inertial = self._link_element(link_name).find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            if origin is not None and origin.get("rpy"):
                com = _rotation(tuple(float(v) for v in origin.get("rpy", "0 0 0").split())) @ com
        if not np.any(inertia):
            inertia = np.asarray([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]) * mass
        if base or joint is None:
            return LinkParams(
                link_name, "base_joint", JointType.FIXED,
                np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1),
                parent_link_name=None, joint_id=0,
                link_mass=mass, link_com=com, link_inertia=inertia,
            )
        limits, joint_kind = self._get_joint_limits(joint)
        axis = np.asarray(joint.axis, dtype=float)
        if joint_kind == "fixed":
            joint_type = JointType.FIXED
            axis_value = None
            joint_limits = None
            velocity_limits = [-2.0, 2.0]
            effort_limit = [10000.0]
            joint_offset = [1.0, 0.0]
        else:
            joint_type = _joint_type(joint_kind, tuple(axis))
            sign = -1.0 if float(axis[np.argmax(np.abs(axis))]) < 0 else 1.0
            axis_value = np.abs(axis)
            joint_limits = [limits["lower"], limits["upper"]]
            velocity_limits = [-limits["velocity"], limits["velocity"]]
            effort_limit = [limits["effort"]]
            joint_offset = [sign * float(joint.mimic_multiplier), float(joint.mimic_offset)]
        transform = np.concatenate(
            [_rotation(joint.rpy), np.asarray(joint.xyz).reshape(3, 1)], axis=1
        )
        return LinkParams(
            link_name=link_name, joint_name=joint.name,
            joint_type=joint_type,
            fixed_transform=transform, parent_link_name=joint.parent,
            child_link_name=joint.child,
            joint_id=self._parent_map[link_name]["jid"],
            joint_limits=joint_limits,
            joint_axis=axis_value,
            joint_velocity_limits=velocity_limits,
            joint_offset=joint_offset,
            mimic_joint_name=joint.mimic_joint,
            joint_effort_limit=effort_limit,
            link_mass=mass, link_com=com, link_inertia=inertia,
        )

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = ""):
        if mesh_dir:
            self.mesh_root = self._mesh_root = mesh_dir
            for link in self._robot.link_map.values():
                for visual in link.visuals:
                    mesh = visual.geometry.mesh
                    if mesh is not None:
                        mesh.filename = self._file_name_handler(mesh.filename)

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
            pose = _pose_from_origin(node.find("origin"))
            name = node.get("name") or f"{link_name}_{tag}_{index}"
            geometry = node.find("geometry")
            if geometry is None:
                continue
            sphere = geometry.find("sphere")
            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            mesh = geometry.find("mesh")
            if sphere is not None:
                result.append(Sphere(
                    name=name, pose=pose,
                    radius=float(sphere.get("radius", "0")),
                ))
            elif box is not None:
                result.append(Cuboid(
                    name=name, pose=pose,
                    dims=[float(v) for v in box.get("size", "0 0 0").split()],
                ))
            elif cylinder is not None:
                result.append(Cylinder(
                    name=name, pose=pose,
                    radius=float(cylinder.get("radius", "0")),
                    height=float(cylinder.get("length", "0")),
                ))
            elif mesh is not None:
                loaded = self.get_link_mesh(link_name, use_collision_mesh)
                if loaded is not None:
                    result.append(loaded)
            else:
                result.append(Obstacle(
                    name=name, pose=pose,
                ))
        return result

    def get_link_mesh(
        self, link_name: str, use_collision_mesh: bool = False
    ) -> Union[CuroboMesh, None]:
        tag = "collision" if use_collision_mesh else "visual"
        # A link with primitive-only or no geometry is not a failure: the V2
        # API returns ``None`` when there is no mesh to materialize.  If a
        # mesh is present, its file-backed construction remains an explicit
        # optional dependency boundary rather than a fabricated empty mesh.
        try:
            nodes = self._link_element(link_name).findall(tag)
        except ValueError as error:
            raise KeyError(link_name) from error
        mesh_node = next(
            (node for node in nodes if node.find("geometry/mesh") is not None), None
        )
        if mesh_node is None:
            return None
        geometry = mesh_node.find("geometry/mesh")
        assert geometry is not None
        file_path = self._file_name_handler(str(geometry.get("filename")))
        source = Path(file_path)
        # Packaged lightweight robot assets may intentionally omit rendering
        # meshes.  Preserve the historical optional result in that case.
        if not source.is_file():
            return None
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
        if source.suffix.lower() == ".stl":
            triangle: list[int] = []
            for line in lines:
                fields = line.strip().split()
                if len(fields) == 4 and fields[0].lower() == "vertex":
                    value = [float(item) for item in fields[1:]]
                    try:
                        index = vertices.index(value)
                    except ValueError:
                        vertices.append(value)
                        index = len(vertices) - 1
                    triangle.append(index)
                    if len(triangle) == 3:
                        faces.append(triangle)
                        triangle = []
        elif source.suffix.lower() == ".obj":
            for line in lines:
                fields = line.strip().split()
                if len(fields) == 4 and fields[0] == "v":
                    vertices.append([float(item) for item in fields[1:]])
                elif len(fields) >= 4 and fields[0] == "f":
                    indices = [int(item.split("/", 1)[0]) - 1 for item in fields[1:]]
                    for index in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[index], indices[index + 1]])
        else:
            return None
        if not vertices or not faces:
            return None
        scale = [float(item) for item in geometry.get("scale", "1 1 1").split()]
        if len(scale) != 3:
            raise ValueError("URDF mesh scale must contain three values")
        values = np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)
        return CuroboMesh(
            name=link_name,
            pose=_pose_from_origin(mesh_node.find("origin")),
            file_path=file_path,
            urdf_path=self.urdf_path,
            vertices=values,
            faces=np.asarray(faces, dtype=np.int64),
        )

    @property
    def root_link(self) -> str:
        return self._robot.base_link

    def get_link_names_from_urdf(self) -> List[str]:
        return [x.name for x in self._robot.links]

    def get_joint_names_from_urdf(self) -> List[str]:
        return [x.name for x in self._robot.joints]

__all__ = ["UrdfRobotParser"]
