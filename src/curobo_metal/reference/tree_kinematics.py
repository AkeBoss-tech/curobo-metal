"""Independent NumPy kinematics for rooted rigid-body trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .forward_kinematics import _motion, _transform

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TreeLink:
    name: str
    parent: int
    kind: str
    axis: FloatArray
    origin: FloatArray
    q_index: int | None
    multiplier: float
    offset: float
    mass: float
    com: FloatArray
    inertia: FloatArray


@dataclass(frozen=True)
class TreeRobot:
    name: str
    links: tuple[TreeLink, ...]
    joint_names: tuple[str, ...]
    end_effectors: tuple[int, ...]
    gravity: FloatArray
    effort_limits: FloatArray

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TreeRobot":
        raw_links = value["links"]
        if not raw_links:
            raise ValueError("robot must contain at least one link")
        names: list[str] = []
        active_names: list[str] = []
        for raw in raw_links:
            name = str(raw["name"])
            if name in names:
                raise ValueError(f"duplicate link name: {name}")
            names.append(name)
            joint = raw.get("joint", {"type": "fixed"})
            if joint.get("type", "fixed") != "fixed" and "mimic" not in joint:
                active_names.append(str(joint.get("name", name)))
        if len(set(active_names)) != len(active_names):
            raise ValueError("active joint names must be unique")
        joint_index = {name: i for i, name in enumerate(active_names)}
        links: list[TreeLink] = []
        for i, raw in enumerate(raw_links):
            parent = int(raw.get("parent", -1))
            if (i == 0 and parent != -1) or (i > 0 and not 0 <= parent < i):
                raise ValueError("links must be topological with one root at index zero")
            joint = raw.get("joint", {"type": "fixed"})
            kind = str(joint.get("type", "fixed"))
            if kind not in {"fixed", "revolute", "prismatic"}:
                raise ValueError(f"unsupported joint type: {kind!r}")
            if i == 0 and kind != "fixed":
                raise ValueError("the root joint must be fixed")
            axis = np.asarray(joint.get("axis", [0, 0, 1]), dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"{names[i]}: invalid joint axis")
            q_index: int | None = None
            multiplier, offset = 1.0, 0.0
            if kind != "fixed":
                norm = float(np.linalg.norm(axis))
                if norm <= 0:
                    raise ValueError(f"{names[i]}: movable joint axis must be nonzero")
                axis = axis / norm
                mimic = joint.get("mimic")
                if mimic is None:
                    q_index = joint_index[str(joint.get("name", names[i]))]
                else:
                    source = str(mimic["joint"])
                    if source not in joint_index:
                        raise ValueError(f"{names[i]}: mimic source must be an active joint")
                    q_index = joint_index[source]
                    multiplier = float(mimic.get("multiplier", 1.0))
                    offset = float(mimic.get("offset", 0.0))
            origin = joint.get("origin", {})
            inertial = raw.get("inertial", {})
            mass = float(inertial.get("mass", 0.0))
            com = np.asarray(inertial.get("com", [0, 0, 0]), dtype=np.float64)
            packed = np.asarray(inertial.get("inertia", [0, 0, 0, 0, 0, 0]), dtype=np.float64)
            if mass < 0 or not np.isfinite(mass) or com.shape != (3,) or packed.shape != (6,):
                raise ValueError(f"{names[i]}: invalid inertial properties")
            inertia = np.array(
                [[packed[0], packed[3], packed[4]], [packed[3], packed[1], packed[5]],
                 [packed[4], packed[5], packed[2]]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(com)) or not np.all(np.isfinite(inertia)):
                raise ValueError(f"{names[i]}: inertial properties must be finite")
            if np.min(np.linalg.eigvalsh(inertia)) < -1e-12:
                raise ValueError(f"{names[i]}: inertia must be positive semidefinite")
            links.append(TreeLink(
                names[i], parent, kind, axis,
                _transform(origin.get("xyz", [0, 0, 0]), origin.get("rpy", [0, 0, 0])),
                q_index, multiplier, offset, mass, com, inertia,
            ))
        end_names = tuple(value.get("end_effectors", []))
        end_effectors = tuple(names.index(str(name)) for name in end_names)
        gravity = np.asarray(value.get("gravity", [0, 0, -9.81]), dtype=np.float64)
        limits = np.asarray(value.get("effort_limits", [np.inf] * len(active_names)), dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must have three finite values")
        if limits.shape != (len(active_names),) or np.any(limits <= 0):
            raise ValueError("effort_limits must be positive and match the degrees of freedom")
        return cls(str(value["name"]), tuple(links), tuple(active_names), end_effectors, gravity, limits)


@dataclass(frozen=True)
class TreeFKResult:
    transforms: FloatArray
    transform_jacobian: FloatArray
    geometric_jacobian: FloatArray
    link_names: tuple[str, ...]
    end_effector_indices: tuple[int, ...]
    input_was_batched: bool


def _batch(value: FloatArray | Sequence[float], width: int, name: str) -> tuple[FloatArray, bool]:
    array = np.asarray(value)
    if array.dtype.kind != "f" or np.iscomplexobj(array):
        raise TypeError(f"{name} must have a real floating-point dtype")
    was_batched = array.ndim == 2
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [{width}] or [B, {width}]")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array, was_batched


def tree_forward_kinematics(
    robot: TreeRobot, q: FloatArray | Sequence[float]
) -> TreeFKResult:
    configurations, was_batched = _batch(q, robot.dof, "q")
    batch, count = configurations.shape[0], len(robot.links)
    transforms = np.empty((batch, count, 4, 4), dtype=np.float64)
    jac = np.zeros((batch, count, 4, 4, robot.dof), dtype=np.float64)
    for b in range(batch):
        for i, link in enumerate(robot.links):
            parent_t = np.eye(4) if link.parent < 0 else transforms[b, link.parent]
            parent_j = np.zeros((4, 4, robot.dof)) if link.parent < 0 else jac[b, link.parent]
            position = 0.0 if link.q_index is None else (
                link.multiplier * configurations[b, link.q_index] + link.offset
            )
            motion, d_motion = _motion(link.kind, link.axis, position)
            local = link.origin @ motion
            transforms[b, i] = parent_t @ local
            for j in range(robot.dof):
                jac[b, i, :, :, j] = parent_j[:, :, j] @ local
                if j == link.q_index:
                    jac[b, i, :, :, j] += parent_t @ link.origin @ d_motion * link.multiplier
    geometric = np.zeros((batch, count, 6, robot.dof), dtype=np.float64)
    for b in range(batch):
        for i in range(count):
            rotation = transforms[b, i, :3, :3]
            geometric[b, i, :3] = jac[b, i, :3, 3]
            for j in range(robot.dof):
                omega = jac[b, i, :3, :3, j] @ rotation.T
                geometric[b, i, 3:, j] = [omega[2, 1], omega[0, 2], omega[1, 0]]
    return TreeFKResult(
        transforms, jac, geometric, tuple(x.name for x in robot.links),
        robot.end_effectors, was_batched,
    )
