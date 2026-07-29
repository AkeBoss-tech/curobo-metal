"""Slow deterministic inverse-kinematics oracle, implemented only with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .costs import (
    CollisionModel,
    joint_limit_cost,
    pose_cost,
    pose_error,
    robot_collision_cost,
)
from .forward_kinematics import SerialRobot, forward_kinematics

FloatArray = NDArray[np.float64]
IK_FORMAT = "curobo-metal-ik-case"
IK_VERSION = 1


@dataclass(frozen=True)
class IKProblem:
    robot: SerialRobot
    target_position: FloatArray
    target_quaternion: FloatArray
    seeds: FloatArray
    lower: FloatArray
    upper: FloatArray
    pose_weights: FloatArray
    position_tolerance: float = 1e-6
    rotation_tolerance: float = 1e-6
    max_iterations: int = 300
    step_tolerance: float = 1e-10
    collision_model: CollisionModel | None = None
    collision_tolerance: float = 0.0
    wrap_revolute: bool = False


@dataclass(frozen=True)
class IKResult:
    solutions: FloatArray
    success: NDArray[np.bool_]
    status: tuple[str, ...]
    iterations: NDArray[np.int64]
    position_error: FloatArray
    rotation_error: FloatArray
    objective: FloatArray
    selected_seed: int | None


def _validate(problem: IKProblem) -> None:
    dof = problem.robot.dof
    if problem.target_position.shape != (3,) or problem.target_quaternion.shape != (4,):
        raise ValueError("target position/quaternion must have shapes [3] and [4]")
    if problem.seeds.ndim != 2 or problem.seeds.shape[1] != dof:
        raise ValueError(f"seeds must have shape [N, {dof}]")
    if problem.lower.shape != (dof,) or problem.upper.shape != (dof,):
        raise ValueError(f"limits must have shape [{dof}]")
    arrays = (
        problem.target_position, problem.target_quaternion, problem.seeds,
        problem.lower, problem.upper, problem.pose_weights,
    )
    if not all(np.asarray(value).dtype.kind == "f" for value in arrays):
        raise TypeError("IK numeric arrays must have floating-point dtypes")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("IK numeric arrays must contain only finite values")
    if np.linalg.norm(problem.target_quaternion) <= 0.0:
        raise ValueError("target quaternion must be nonzero")
    if np.any(problem.lower > problem.upper):
        raise ValueError("joint limits must satisfy lower <= upper")
    if problem.pose_weights.shape != (6,) or np.any(problem.pose_weights < 0):
        raise ValueError("pose_weights must be nonnegative with shape [6]")
    if problem.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    tolerances = (
        problem.position_tolerance, problem.rotation_tolerance,
        problem.step_tolerance, problem.collision_tolerance,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in tolerances):
        raise ValueError("IK tolerances must be finite and nonnegative")


def _wrap_and_project(problem: IKProblem, q: FloatArray) -> FloatArray:
    result = q.copy()
    if problem.wrap_revolute:
        for joint in problem.robot.joints:
            if joint.q_index is not None and joint.kind == "revolute":
                index = joint.q_index
                result[index] = (result[index] + np.pi) % (2 * np.pi) - np.pi
    return np.clip(result, problem.lower, problem.upper)


def _evaluate(problem: IKProblem, q: FloatArray) -> tuple[float, FloatArray, float]:
    transform = forward_kinematics(problem.robot, q).transforms[0, -1]
    residual = pose_error(transform, problem.target_position, problem.target_quaternion)
    value = pose_cost(residual, problem.pose_weights).value
    value += joint_limit_cost(q, problem.lower, problem.upper, weight=1e3).value
    collision = 0.0
    if problem.collision_model is not None:
        collision = robot_collision_cost(problem.robot, q, problem.collision_model)
        value += collision
    return value, residual, collision


def _finite_difference_gradient(problem: IKProblem, q: FloatArray, step: float = 1e-6) -> FloatArray:
    gradient = np.empty_like(q)
    for index in range(q.size):
        plus, minus = q.copy(), q.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (_evaluate(problem, plus)[0] - _evaluate(problem, minus)[0]) / (2 * step)
    return gradient


def solve_ik(problem: IKProblem) -> IKResult:
    """Solve every seed independently with deterministic projected descent."""
    _validate(problem)
    count, dof = problem.seeds.shape
    solutions = np.empty((count, dof), dtype=np.float64)
    successes = np.zeros(count, dtype=np.bool_)
    statuses: list[str] = []
    iterations = np.zeros(count, dtype=np.int64)
    positions = np.empty(count, dtype=np.float64)
    rotations = np.empty(count, dtype=np.float64)
    objectives = np.empty(count, dtype=np.float64)
    for seed_index, seed in enumerate(problem.seeds):
        q = _wrap_and_project(problem, np.asarray(seed, dtype=np.float64))
        status = "max_iterations"
        for iteration in range(1, problem.max_iterations + 1):
            value, residual, collision = _evaluate(problem, q)
            position_norm = float(np.linalg.norm(residual[:3]))
            rotation_norm = float(np.linalg.norm(residual[3:]))
            if (position_norm <= problem.position_tolerance
                    and rotation_norm <= problem.rotation_tolerance
                    and collision <= problem.collision_tolerance):
                status = "success"
                successes[seed_index] = True
                break
            gradient = _finite_difference_gradient(problem, q)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= problem.step_tolerance:
                status = "infeasible_or_stationary"
                break
            direction = -gradient
            slope = float(gradient @ direction)
            step_size = min(1.0, 0.25 / gradient_norm)
            accepted = False
            for _ in range(30):
                candidate = _wrap_and_project(problem, q + step_size * direction)
                candidate_value = _evaluate(problem, candidate)[0]
                if candidate_value <= value + 1e-4 * step_size * slope:
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted or np.linalg.norm(candidate - q) <= problem.step_tolerance:
                status = "line_search_failed"
                break
            q = candidate
        final_value, final_residual, final_collision = _evaluate(problem, q)
        # Categorize failures using observable contract conditions.
        if not successes[seed_index]:
            if final_collision > problem.collision_tolerance:
                status = "collision_constrained"
            elif np.any(np.isclose(q, problem.lower, atol=1e-9)) or np.any(
                np.isclose(q, problem.upper, atol=1e-9)
            ):
                status = "limit_constrained"
        solutions[seed_index] = q
        statuses.append(status)
        iterations[seed_index] = iteration
        positions[seed_index] = np.linalg.norm(final_residual[:3])
        rotations[seed_index] = np.linalg.norm(final_residual[3:])
        objectives[seed_index] = final_value
    successful = np.flatnonzero(successes)
    selected = (
        int(successful[np.argmin(objectives[successful])])
        if successful.size else None
    )
    return IKResult(
        solutions, successes, tuple(statuses), iterations, positions, rotations,
        objectives, selected,
    )


def problem_from_dict(value: Mapping[str, Any]) -> IKProblem:
    robot = SerialRobot.from_dict(value["robot"])
    inputs, options = value["inputs"], value.get("options", {})
    collision = inputs.get("collision")
    model = None
    if collision is not None:
        model = CollisionModel(
            local_spheres=np.asarray(collision["local_spheres"], dtype=np.float64),
            link_indices=np.asarray(collision["link_indices"], dtype=np.int64),
            self_pairs=(None if "self_pairs" not in collision else
                        np.asarray(collision["self_pairs"], dtype=np.int64)),
            cuboid_centers=(None if "cuboid_centers" not in collision else
                            np.asarray(collision["cuboid_centers"], dtype=np.float64)),
            cuboid_rotations=(None if "cuboid_rotations" not in collision else
                              np.asarray(collision["cuboid_rotations"], dtype=np.float64)),
            cuboid_half_extents=(None if "cuboid_half_extents" not in collision else
                                 np.asarray(collision["cuboid_half_extents"], dtype=np.float64)),
            padding=float(collision.get("padding", 0.0)),
            activation_distance=float(collision.get("activation_distance", 0.0)),
            weight=float(collision.get("weight", 1.0)),
        )
    return IKProblem(
        robot=robot,
        target_position=np.asarray(inputs["target_position"], dtype=np.float64),
        target_quaternion=np.asarray(inputs["target_quaternion"], dtype=np.float64),
        seeds=np.asarray(inputs["seeds"], dtype=np.float64),
        lower=np.asarray(inputs["lower"], dtype=np.float64),
        upper=np.asarray(inputs["upper"], dtype=np.float64),
        pose_weights=np.asarray(inputs.get("pose_weights", np.ones(6)), dtype=np.float64),
        position_tolerance=float(options.get("position_tolerance", 1e-6)),
        rotation_tolerance=float(options.get("rotation_tolerance", 1e-6)),
        max_iterations=int(options.get("max_iterations", 300)),
        step_tolerance=float(options.get("step_tolerance", 1e-10)),
        collision_model=model,
        collision_tolerance=float(options.get("collision_tolerance", 0.0)),
        wrap_revolute=bool(options.get("wrap_revolute", False)),
    )


def load_ik_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("format") != IK_FORMAT or value.get("version") != IK_VERSION:
        raise ValueError("unsupported IK replay format")
    return value


def save_ik_case(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_ik_case(destination)
