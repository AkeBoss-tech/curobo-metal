"""Deterministic NumPy trajectory-optimization and motion-generation oracle."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .collision import sphere_cuboid_signed_distance, sphere_sphere_signed_distance, transform_spheres
from .costs import CollisionModel, ScalarCost, joint_limit_cost, robot_collision_cost, smoothness_cost
from .forward_kinematics import SerialRobot, forward_kinematics

FloatArray = NDArray[np.float64]
TRAJECTORY_FORMAT = "curobo-metal-trajectory-case"
TRAJECTORY_VERSION = 1


def _float(value: Any, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real floating point")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def minimum_jerk_trajectory(start: Any, goal: Any, steps: int) -> FloatArray:
    """Return the zero endpoint velocity/acceleration quintic interpolation."""
    q0, q1 = _float(start, "start"), _float(goal, "goal")
    if q0.ndim != 1 or q1.shape != q0.shape:
        raise ValueError("start and goal must have equal shape [J]")
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_)) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    phase = np.linspace(0.0, 1.0, int(steps), dtype=np.float64)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return q0 + blend[:, None] * (q1 - q0)


def endpoint_cost(q: Any, start: Any, goal: Any, *, weight: float = 1.0) -> ScalarCost:
    trajectory = _float(q, "q")
    q0, q1 = _float(start, "start"), _float(goal, "goal")
    if trajectory.ndim != 2 or q0.shape != (trajectory.shape[1],) or q1.shape != q0.shape:
        raise ValueError("q, start, and goal must have shapes [T,J], [J], and [J]")
    if trajectory.shape[0] < 2:
        raise ValueError("trajectory must contain at least two knots")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    gradient = np.zeros_like(trajectory)
    gradient[0] = weight * (trajectory[0] - q0)
    gradient[-1] = weight * (trajectory[-1] - q1)
    value = 0.5 * weight * (
        float(np.sum((trajectory[0] - q0) ** 2))
        + float(np.sum((trajectory[-1] - q1) ** 2))
    )
    return ScalarCost(value, gradient)


def interpolated_states(q: Any, subdivisions: int = 1) -> tuple[FloatArray, FloatArray]:
    """Sample all knots and equally spaced swept states without duplicates.

    Returns states ``[1 + (T-1)*subdivisions,J]`` and knot coordinates, where
    coordinate ``i + f`` identifies linear interpolation in segment ``i``.
    """
    trajectory = _float(q, "q")
    if trajectory.ndim != 2 or trajectory.shape[0] < 1:
        raise ValueError("q must have shape [T,J] with T >= 1")
    if (not isinstance(subdivisions, (int, np.integer))
            or isinstance(subdivisions, (bool, np.bool_)) or subdivisions < 1):
        raise ValueError("subdivisions must be a positive integer")
    if trajectory.shape[0] == 1:
        return trajectory.copy(), np.array([0.0])
    coordinates = np.arange((trajectory.shape[0] - 1) * subdivisions + 1) / subdivisions
    low = np.minimum(coordinates.astype(np.int64), trajectory.shape[0] - 2)
    fraction = coordinates - low
    fraction[-1] = 1.0
    states = (1.0 - fraction[:, None]) * trajectory[low] + fraction[:, None] * trajectory[low + 1]
    return states, coordinates


def _clearances(robot: SerialRobot, states: FloatArray, model: CollisionModel) -> FloatArray:
    transforms = forward_kinematics(robot, states).transforms
    spheres = transform_spheres(transforms, model.local_spheres, model.link_indices).spheres
    values: list[FloatArray] = []
    if model.self_pairs is not None:
        result = sphere_sphere_signed_distance(spheres, model.self_pairs, padding=model.padding)
        values.append(result.distances)
    if model.cuboid_centers is not None:
        result = sphere_cuboid_signed_distance(
            spheres, model.cuboid_centers, model.cuboid_rotations,
            model.cuboid_half_extents, padding=model.padding,
        )
        values.append(result.distances.reshape(states.shape[0], -1))
    if not values:
        return np.empty((states.shape[0], 0), dtype=np.float64)
    return np.concatenate(values, axis=1)


def trajectory_collision_cost(
    robot: SerialRobot,
    q: Any,
    model: CollisionModel,
    *,
    subdivisions: int = 1,
    finite_difference_step: float = 1e-6,
) -> tuple[ScalarCost, FloatArray, FloatArray]:
    """Collision hinge over discrete and swept states with a reference gradient."""
    trajectory = _float(q, "q")
    states, coordinates = interpolated_states(trajectory, subdivisions)
    clearances = _clearances(robot, states, model)
    value = sum(robot_collision_cost(robot, state, model) for state in states)
    state_gradient = np.zeros_like(states)
    if model.weight != 0.0:
        for row, state in enumerate(states):
            for joint in range(state.size):
                plus, minus = state.copy(), state.copy()
                plus[joint] += finite_difference_step
                minus[joint] -= finite_difference_step
                state_gradient[row, joint] = (
                    robot_collision_cost(robot, plus, model)
                    - robot_collision_cost(robot, minus, model)
                ) / (2.0 * finite_difference_step)
    gradient = np.zeros_like(trajectory)
    for row, coordinate in enumerate(coordinates):
        low = min(int(np.floor(coordinate)), trajectory.shape[0] - 2)
        fraction = coordinate - low
        if row == len(coordinates) - 1:
            fraction = 1.0
        gradient[low] += (1.0 - fraction) * state_gradient[row]
        gradient[low + 1] += fraction * state_gradient[row]
    return ScalarCost(float(value), gradient), clearances, coordinates


@dataclass(frozen=True)
class TrajectoryWeights:
    endpoint: float = 1_000.0
    joint_limit: float = 10.0
    velocity: float = 0.05
    acceleration: float = 1.0
    jerk: float = 0.05


@dataclass(frozen=True)
class TrajectoryProblem:
    robot: SerialRobot
    start: FloatArray
    goal: FloatArray
    lower: FloatArray
    upper: FloatArray
    steps: int
    dt: float
    seeds: FloatArray | None = None
    weights: TrajectoryWeights = TrajectoryWeights()
    collision_model: CollisionModel | None = None
    collision_subdivisions: int = 1
    endpoint_tolerance: float = 1e-8
    collision_tolerance: float = 0.0
    max_iterations: int = 120
    gradient_tolerance: float = 1e-8
    finite_difference_step: float = 1e-6


@dataclass(frozen=True)
class TrajectoryCost:
    value: float
    gradient: FloatArray
    endpoint: float
    joint_limit: float
    smoothness: float
    collision: float
    minimum_clearance: float


@dataclass(frozen=True)
class TrajectoryResult:
    trajectories: FloatArray
    success: NDArray[np.bool_]
    status: tuple[str, ...]
    iterations: NDArray[np.int64]
    objective: FloatArray
    endpoint_error: FloatArray
    minimum_clearance: FloatArray
    selected_seed: int | None


def _validate(problem: TrajectoryProblem) -> None:
    dof = problem.robot.dof
    arrays = (problem.start, problem.goal, problem.lower, problem.upper)
    if any(np.asarray(value).shape != (dof,) for value in arrays):
        raise ValueError(f"start, goal, lower, and upper must have shape [{dof}]")
    if any(np.asarray(value).dtype.kind != "f" for value in arrays):
        raise TypeError("trajectory numeric arrays must have floating-point dtypes")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("trajectory numeric arrays must be finite")
    if np.any(problem.lower > problem.upper):
        raise ValueError("joint limits must satisfy lower <= upper")
    if not isinstance(problem.steps, (int, np.integer)) or problem.steps < 2:
        raise ValueError("steps must be an integer >= 2")
    if problem.seeds is not None:
        seed_array = np.asarray(problem.seeds)
        if seed_array.dtype.kind != "f":
            raise TypeError("seeds must have floating-point dtype")
        if seed_array.ndim != 3 or seed_array.shape[1:] != (problem.steps, dof):
            raise ValueError(f"seeds must have shape [N,{problem.steps},{dof}]")
        if not np.all(np.isfinite(seed_array)):
            raise ValueError("seeds must be finite")
    scalars = (
        problem.dt, problem.endpoint_tolerance, problem.collision_tolerance,
        problem.gradient_tolerance, problem.finite_difference_step,
    )
    if (not np.isfinite(problem.dt) or problem.dt <= 0.0
            or any(not np.isfinite(x) or x < 0.0 for x in scalars[1:])
            or problem.finite_difference_step <= 0.0):
        raise ValueError("dt/finite-difference step must be positive and tolerances nonnegative")
    if problem.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    interpolated_states(np.zeros((2, dof), dtype=np.float64), problem.collision_subdivisions)
    if any(not np.isfinite(x) or x < 0.0 for x in vars(problem.weights).values()):
        raise ValueError("trajectory weights must be finite and nonnegative")


def evaluate_trajectory(problem: TrajectoryProblem, q: Any) -> TrajectoryCost:
    _validate(problem)
    trajectory = _float(q, "q")
    if trajectory.shape != (problem.steps, problem.robot.dof):
        raise ValueError(f"q must have shape [{problem.steps},{problem.robot.dof}]")
    end = endpoint_cost(trajectory, problem.start, problem.goal, weight=problem.weights.endpoint)
    limit = joint_limit_cost(
        trajectory,
        np.broadcast_to(problem.lower, trajectory.shape),
        np.broadcast_to(problem.upper, trajectory.shape),
        weight=problem.weights.joint_limit,
    )
    smooth = smoothness_cost(
        trajectory, velocity_weight=problem.weights.velocity,
        acceleration_weight=problem.weights.acceleration,
        jerk_weight=problem.weights.jerk, dt=problem.dt,
    )
    collision = ScalarCost(0.0, np.zeros_like(trajectory))
    minimum_clearance = np.inf
    if problem.collision_model is not None:
        collision, clearances, _ = trajectory_collision_cost(
            problem.robot, trajectory, problem.collision_model,
            subdivisions=problem.collision_subdivisions,
            finite_difference_step=problem.finite_difference_step,
        )
        if clearances.size:
            minimum_clearance = float(np.min(clearances))
    return TrajectoryCost(
        end.value + limit.value + smooth.value + collision.value,
        end.gradient + limit.gradient + smooth.gradient + collision.gradient,
        end.value, limit.value, smooth.value, collision.value, minimum_clearance,
    )


def _project(problem: TrajectoryProblem, q: FloatArray) -> FloatArray:
    result = np.clip(q, problem.lower, problem.upper)
    result[0] = np.clip(problem.start, problem.lower, problem.upper)
    result[-1] = np.clip(problem.goal, problem.lower, problem.upper)
    return result


def optimize_trajectory(problem: TrajectoryProblem) -> TrajectoryResult:
    """Solve each seed independently using projected gradient and backtracking."""
    _validate(problem)
    seeds = (
        minimum_jerk_trajectory(problem.start, problem.goal, problem.steps)[None]
        if problem.seeds is None else np.asarray(problem.seeds, dtype=np.float64)
    )
    count, _, dof = seeds.shape
    trajectories = np.empty((count, problem.steps, dof), dtype=np.float64)
    success = np.zeros(count, dtype=np.bool_)
    statuses: list[str] = []
    iterations = np.zeros(count, dtype=np.int64)
    objectives = np.empty(count)
    endpoint_errors = np.empty(count)
    clearances = np.empty(count)
    endpoints_feasible = (
        np.all(problem.start >= problem.lower) and np.all(problem.start <= problem.upper)
        and np.all(problem.goal >= problem.lower) and np.all(problem.goal <= problem.upper)
    )
    for seed_index, seed in enumerate(seeds):
        q = _project(problem, seed.copy())
        status = "max_iterations"
        iteration = 0
        if endpoints_feasible:
            for iteration in range(1, problem.max_iterations + 1):
                current = evaluate_trajectory(problem, q)
                gradient = current.gradient
                gradient[[0, -1]] = 0.0
                norm = float(np.linalg.norm(gradient))
                if norm <= problem.gradient_tolerance:
                    status = "stationary"
                    break
                direction = -gradient
                slope = float(np.sum(gradient * direction))
                step_size = min(1.0, 0.2 / norm)
                accepted = False
                for _ in range(30):
                    candidate = _project(problem, q + step_size * direction)
                    candidate_value = evaluate_trajectory(problem, candidate).value
                    if candidate_value <= current.value + 1e-4 * step_size * slope:
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted or np.linalg.norm(candidate - q) <= problem.gradient_tolerance:
                    status = "line_search_failed"
                    break
                q = candidate
        else:
            status = "endpoint_infeasible"
        final = evaluate_trajectory(problem, q)
        endpoint_error = max(
            float(np.linalg.norm(q[0] - problem.start)),
            float(np.linalg.norm(q[-1] - problem.goal)),
        )
        collision_ok = final.minimum_clearance >= -problem.collision_tolerance
        if endpoints_feasible and endpoint_error <= problem.endpoint_tolerance and collision_ok:
            success[seed_index] = True
            status = "success"
        elif not collision_ok:
            status = "collision_constrained"
        trajectories[seed_index] = q
        statuses.append(status)
        iterations[seed_index] = iteration
        objectives[seed_index] = final.value
        endpoint_errors[seed_index] = endpoint_error
        clearances[seed_index] = final.minimum_clearance
    candidates = np.flatnonzero(success)
    selected = int(candidates[np.argmin(objectives[candidates])]) if candidates.size else None
    return TrajectoryResult(
        trajectories, success, tuple(statuses), iterations, objectives,
        endpoint_errors, clearances, selected,
    )


def _collision_from_dict(value: Mapping[str, Any] | None) -> CollisionModel | None:
    if value is None:
        return None
    return CollisionModel(
        local_spheres=np.asarray(value["local_spheres"], dtype=np.float64),
        link_indices=np.asarray(value["link_indices"], dtype=np.int64),
        self_pairs=(None if "self_pairs" not in value else np.asarray(value["self_pairs"], dtype=np.int64)),
        cuboid_centers=(None if "cuboid_centers" not in value else np.asarray(value["cuboid_centers"], dtype=np.float64)),
        cuboid_rotations=(None if "cuboid_rotations" not in value else np.asarray(value["cuboid_rotations"], dtype=np.float64)),
        cuboid_half_extents=(None if "cuboid_half_extents" not in value else np.asarray(value["cuboid_half_extents"], dtype=np.float64)),
        padding=float(value.get("padding", 0.0)),
        activation_distance=float(value.get("activation_distance", 0.0)),
        weight=float(value.get("weight", 1.0)),
    )


def trajectory_problem_from_dict(value: Mapping[str, Any]) -> TrajectoryProblem:
    inputs, options = value["inputs"], value.get("options", {})
    weights = options.get("weights", {})
    seeds = inputs.get("seeds")
    return TrajectoryProblem(
        robot=SerialRobot.from_dict(value["robot"]),
        start=np.asarray(inputs["start"], dtype=np.float64),
        goal=np.asarray(inputs["goal"], dtype=np.float64),
        lower=np.asarray(inputs["lower"], dtype=np.float64),
        upper=np.asarray(inputs["upper"], dtype=np.float64),
        steps=int(inputs["steps"]), dt=float(inputs["dt"]),
        seeds=None if seeds is None else np.asarray(seeds, dtype=np.float64),
        weights=TrajectoryWeights(**{key: float(item) for key, item in weights.items()}),
        collision_model=_collision_from_dict(inputs.get("collision")),
        collision_subdivisions=int(options.get("collision_subdivisions", 1)),
        endpoint_tolerance=float(options.get("endpoint_tolerance", 1e-8)),
        collision_tolerance=float(options.get("collision_tolerance", 0.0)),
        max_iterations=int(options.get("max_iterations", 120)),
        gradient_tolerance=float(options.get("gradient_tolerance", 1e-8)),
        finite_difference_step=float(options.get("finite_difference_step", 1e-6)),
    )


def load_trajectory_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("format") != TRAJECTORY_FORMAT or value.get("version") != TRAJECTORY_VERSION:
        raise ValueError("unsupported trajectory replay format")
    return value


def save_trajectory_case(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_trajectory_case(destination)
