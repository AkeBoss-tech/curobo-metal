"""Deterministic rigid-body dynamics reference for fixed-base trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .tree_kinematics import FloatArray, TreeRobot, _batch, tree_forward_kinematics


@dataclass(frozen=True)
class DynamicsResult:
    torque: FloatArray
    input_was_batched: bool


@dataclass(frozen=True)
class DynamicsCost:
    total: FloatArray
    effort: FloatArray
    limit: FloatArray


@dataclass(frozen=True)
class DynamicsDerivatives:
    d_tau_d_q: FloatArray
    d_tau_d_qd: FloatArray
    d_tau_d_qdd: FloatArray


def inverse_dynamics(
    robot: TreeRobot,
    q: FloatArray | Sequence[float],
    qd: FloatArray | Sequence[float],
    qdd: FloatArray | Sequence[float],
    *,
    gravity: FloatArray | Sequence[float] | None = None,
) -> DynamicsResult:
    """Recursive Newton-Euler inverse dynamics in world coordinates."""
    q_b, was_batched = _batch(q, robot.dof, "q")
    qd_b, _ = _batch(qd, robot.dof, "qd")
    qdd_b, _ = _batch(qdd, robot.dof, "qdd")
    if q_b.shape != qd_b.shape or q_b.shape != qdd_b.shape:
        raise ValueError("q, qd, and qdd batch dimensions must match")
    g = robot.gravity if gravity is None else np.asarray(gravity, dtype=np.float64)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError("gravity must have shape [3] and be finite")
    fk = tree_forward_kinematics(robot, q_b)
    batch, count = q_b.shape[0], len(robot.links)
    torque = np.zeros((batch, robot.dof), dtype=np.float64)
    for b in range(batch):
        omega = np.zeros((count, 3)); alpha = np.zeros((count, 3))
        velocity = np.zeros((count, 3)); acceleration = np.zeros((count, 3))
        force = np.zeros((count, 3)); moment = np.zeros((count, 3))
        axes = np.zeros((count, 3))
        for i, link in enumerate(robot.links):
            p = link.parent
            wp = np.zeros(3) if p < 0 else omega[p]
            ap = np.zeros(3) if p < 0 else alpha[p]
            vp = np.zeros(3) if p < 0 else velocity[p]
            alp = np.zeros(3) if p < 0 else acceleration[p]
            parent_pos = np.zeros(3) if p < 0 else fk.transforms[b, p, :3, 3]
            r = fk.transforms[b, i, :3, 3] - parent_pos
            velocity[i] = vp + np.cross(wp, r)
            acceleration[i] = alp + np.cross(ap, r) + np.cross(wp, np.cross(wp, r))
            qi = link.q_index
            qdi = 0.0 if qi is None else link.multiplier * qd_b[b, qi]
            qddi = 0.0 if qi is None else link.multiplier * qdd_b[b, qi]
            # Joint axis is expressed after the fixed origin, before joint motion.
            parent_r = np.eye(3) if p < 0 else fk.transforms[b, p, :3, :3]
            axis = parent_r @ link.origin[:3, :3] @ link.axis
            axes[i] = axis
            if link.kind == "revolute":
                omega[i] = wp + axis * qdi
                alpha[i] = ap + axis * qddi + np.cross(wp, axis * qdi)
            else:
                omega[i], alpha[i] = wp, ap
                if link.kind == "prismatic":
                    velocity[i] += axis * qdi
                    acceleration[i] += axis * qddi + 2 * np.cross(wp, axis * qdi)
            rotation = fk.transforms[b, i, :3, :3]
            c = rotation @ link.com
            com_acc = acceleration[i] + np.cross(alpha[i], c) + np.cross(
                omega[i], np.cross(omega[i], c)
            )
            force[i] = link.mass * (com_acc - g)
            inertia = rotation @ link.inertia @ rotation.T
            moment[i] = inertia @ alpha[i] + np.cross(omega[i], inertia @ omega[i])
            moment[i] += np.cross(c, force[i])
        for i in range(count - 1, -1, -1):
            link = robot.links[i]
            if link.q_index is not None:
                effort = (np.dot(axes[i], moment[i]) if link.kind == "revolute"
                          else np.dot(axes[i], force[i]))
                torque[b, link.q_index] += link.multiplier * effort
            if link.parent >= 0:
                p = link.parent
                displacement = (
                    fk.transforms[b, i, :3, 3] - fk.transforms[b, p, :3, 3]
                )
                force[p] += force[i]
                moment[p] += moment[i] + np.cross(displacement, force[i])
    return DynamicsResult(torque, was_batched)


def mass_matrix(robot: TreeRobot, q: FloatArray | Sequence[float]) -> FloatArray:
    q_b, _ = _batch(q, robot.dof, "q")
    zeros = np.zeros_like(q_b)
    matrix = np.empty((q_b.shape[0], robot.dof, robot.dof), dtype=np.float64)
    for j in range(robot.dof):
        unit = np.zeros_like(q_b)
        unit[:, j] = 1.0
        matrix[:, :, j] = inverse_dynamics(
            robot, q_b, zeros, unit, gravity=np.zeros(3)
        ).torque
    return 0.5 * (matrix + matrix.transpose(0, 2, 1))


def gravity_torque(robot: TreeRobot, q: FloatArray | Sequence[float]) -> FloatArray:
    q_b, _ = _batch(q, robot.dof, "q")
    zeros = np.zeros_like(q_b)
    return inverse_dynamics(robot, q_b, zeros, zeros).torque


def bias_torque(
    robot: TreeRobot, q: FloatArray | Sequence[float], qd: FloatArray | Sequence[float]
) -> FloatArray:
    q_b, _ = _batch(q, robot.dof, "q")
    return inverse_dynamics(robot, q_b, qd, np.zeros_like(q_b)).torque


def dynamics_cost(
    robot: TreeRobot,
    torque: FloatArray | Sequence[float],
    *,
    effort_weight: float = 1.0,
    limit_weight: float = 1.0,
) -> DynamicsCost:
    values, _ = _batch(torque, robot.dof, "torque")
    effort = effort_weight * np.sum(values * values, axis=1)
    violation = np.maximum(np.abs(values) - robot.effort_limits, 0.0)
    limit = limit_weight * np.sum(violation * violation, axis=1)
    return DynamicsCost(effort + limit, effort, limit)


def inverse_dynamics_derivatives(
    robot: TreeRobot,
    q: FloatArray | Sequence[float],
    qd: FloatArray | Sequence[float],
    qdd: FloatArray | Sequence[float],
    *,
    step: float = 1e-6,
) -> DynamicsDerivatives:
    """Central-finite-difference torque Jacobians for backend gradient replay."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    values = [_batch(x, robot.dof, name)[0] for x, name in (
        (q, "q"), (qd, "qd"), (qdd, "qdd")
    )]
    if values[0].shape != values[1].shape or values[0].shape != values[2].shape:
        raise ValueError("q, qd, and qdd batch dimensions must match")
    outputs = [
        np.empty((values[0].shape[0], robot.dof, robot.dof), dtype=np.float64)
        for _ in range(3)
    ]
    for argument in range(3):
        for j in range(robot.dof):
            plus = [x.copy() for x in values]
            minus = [x.copy() for x in values]
            plus[argument][:, j] += step
            minus[argument][:, j] -= step
            outputs[argument][:, :, j] = (
                inverse_dynamics(robot, *plus).torque
                - inverse_dynamics(robot, *minus).torque
            ) / (2 * step)
    return DynamicsDerivatives(*outputs)
