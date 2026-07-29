"""Deterministic NumPy forward kinematics for serial robots.

This module deliberately has no dependency on PyTorch, cuRobo, MPS, or CUDA.
It is an executable specification, not a performance implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _skew(axis: FloatArray) -> FloatArray:
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rpy_matrix(rpy: Sequence[float]) -> FloatArray:
    """URDF fixed-axis roll-pitch-yaw rotation, Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> FloatArray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = xyz
    return out


def _motion(kind: str, axis: FloatArray, q: float) -> tuple[FloatArray, FloatArray]:
    """Return motion transform and its exact derivative with respect to q."""
    transform = np.eye(4, dtype=np.float64)
    derivative = np.zeros((4, 4), dtype=np.float64)
    if kind == "fixed":
        return transform, derivative
    if kind == "prismatic":
        transform[:3, 3] = axis * q
        derivative[:3, 3] = axis
        return transform, derivative
    if kind != "revolute":
        raise ValueError(f"unsupported joint type: {kind!r}")
    k = _skew(axis)
    rotation = np.eye(3) + np.sin(q) * k + (1.0 - np.cos(q)) * (k @ k)
    rotation_derivative = np.cos(q) * k + np.sin(q) * (k @ k)
    transform[:3, :3] = rotation
    derivative[:3, :3] = rotation_derivative
    return transform, derivative


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    axis: FloatArray
    origin: FloatArray
    q_index: int | None


@dataclass(frozen=True)
class SerialRobot:
    """Validated immutable serial-chain description."""

    name: str
    joints: tuple[Joint, ...]
    dof: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SerialRobot":
        joints: list[Joint] = []
        q_index = 0
        seen_names: set[str] = set()
        for raw in value["joints"]:
            name = str(raw["name"])
            if name in seen_names:
                raise ValueError(f"duplicate joint name: {name}")
            seen_names.add(name)
            kind = str(raw["type"])
            if kind not in {"revolute", "prismatic", "fixed"}:
                raise ValueError(f"unsupported joint type: {kind!r}")
            axis = np.asarray(raw.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"{name}: axis must contain three finite values")
            index: int | None = None
            if kind != "fixed":
                norm = float(np.linalg.norm(axis))
                if norm <= 0.0:
                    raise ValueError(f"{name}: movable joint axis must be nonzero")
                axis = axis / norm
                index = q_index
                q_index += 1
            origin = raw.get("origin", {})
            joints.append(
                Joint(
                    name=name,
                    kind=kind,
                    axis=axis,
                    origin=_transform(
                        origin.get("xyz", [0.0, 0.0, 0.0]),
                        origin.get("rpy", [0.0, 0.0, 0.0]),
                    ),
                    q_index=index,
                )
            )
        if not joints:
            raise ValueError("robot must contain at least one joint/link")
        return cls(name=str(value["name"]), joints=tuple(joints), dof=q_index)


@dataclass(frozen=True)
class FKResult:
    """All arrays are float64 and own their storage."""

    transforms: FloatArray  # [B, L, 4, 4]
    transform_jacobian: FloatArray  # [B, L, 4, 4, J]
    geometric_jacobian: FloatArray  # [B, L, 6, J], linear then angular
    link_names: tuple[str, ...]
    input_was_batched: bool


def forward_kinematics(robot: SerialRobot, q: FloatArray | Sequence[float]) -> FKResult:
    """Evaluate a batch of serial-chain configurations deterministically."""
    configurations = np.asarray(q)
    if configurations.dtype.kind not in "fc":
        raise TypeError("q must have a floating-point dtype")
    if np.iscomplexobj(configurations):
        raise TypeError("q must be real")
    input_was_batched = configurations.ndim == 2
    if configurations.ndim == 1:
        configurations = configurations[None, :]
    if configurations.ndim != 2 or configurations.shape[1] != robot.dof:
        raise ValueError(f"q must have shape [{robot.dof}] or [B, {robot.dof}]")
    configurations = np.asarray(configurations, dtype=np.float64)
    if not np.all(np.isfinite(configurations)):
        raise ValueError("q must contain only finite values")

    batch, links = configurations.shape[0], len(robot.joints)
    transforms = np.empty((batch, links, 4, 4), dtype=np.float64)
    derivatives = np.zeros((batch, links, 4, 4, robot.dof), dtype=np.float64)

    for b in range(batch):
        current = np.eye(4, dtype=np.float64)
        current_derivatives = np.zeros((4, 4, robot.dof), dtype=np.float64)
        for link_index, joint in enumerate(robot.joints):
            position = 0.0 if joint.q_index is None else configurations[b, joint.q_index]
            motion, motion_derivative = _motion(joint.kind, joint.axis, position)
            local = joint.origin @ motion
            local_derivative = joint.origin @ motion_derivative
            next_derivatives = np.empty_like(current_derivatives)
            for column in range(robot.dof):
                next_derivatives[:, :, column] = current_derivatives[:, :, column] @ local
                if column == joint.q_index:
                    next_derivatives[:, :, column] += current @ local_derivative
            current = current @ local
            current_derivatives = next_derivatives
            transforms[b, link_index] = current
            derivatives[b, link_index] = current_derivatives

    geometric = np.zeros((batch, links, 6, robot.dof), dtype=np.float64)
    for b in range(batch):
        for link_index in range(links):
            rotation = transforms[b, link_index, :3, :3]
            position = transforms[b, link_index, :3, 3]
            for column in range(robot.dof):
                d_rotation = derivatives[b, link_index, :3, :3, column]
                geometric[b, link_index, :3, column] = derivatives[
                    b, link_index, :3, 3, column
                ]
                # vee(dR R^T), robust for both revolute and prismatic columns.
                omega = d_rotation @ rotation.T
                geometric[b, link_index, 3:, column] = [
                    omega[2, 1],
                    omega[0, 2],
                    omega[1, 0],
                ]

    return FKResult(
        transforms=transforms,
        transform_jacobian=derivatives,
        geometric_jacobian=geometric,
        link_names=tuple(joint.name for joint in robot.joints),
        input_was_batched=input_was_batched,
    )

