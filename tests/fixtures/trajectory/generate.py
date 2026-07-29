"""Regenerate canonical trajectory replay fixtures and correctness evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from curobo_metal.reference import (
    CollisionModel,
    SerialRobot,
    TrajectoryProblem,
    TrajectoryWeights,
    load_case,
    optimize_trajectory,
)

ROOT = Path(__file__).parents[3]
FIXTURES = Path(__file__).parent
ARTIFACT = ROOT / "artifacts" / "correctness" / "trajectory_reference.json"


def _robot(name: str) -> tuple[SerialRobot, dict[str, Any]]:
    value = load_case(ROOT / "tests" / "fixtures" / name)["robot"]
    return SerialRobot.from_dict(value), value


def _collision_dict(model: CollisionModel) -> dict[str, Any]:
    result: dict[str, Any] = {
        "activation_distance": model.activation_distance,
        "link_indices": model.link_indices.tolist(),
        "local_spheres": model.local_spheres.tolist(),
        "padding": model.padding,
        "weight": model.weight,
    }
    for name in ("self_pairs", "cuboid_centers", "cuboid_rotations", "cuboid_half_extents"):
        item = getattr(model, name)
        if item is not None:
            result[name] = item.tolist()
    return result


def _case(
    name: str,
    robot_dict: dict[str, Any],
    problem: TrajectoryProblem,
    description: str,
) -> dict[str, Any]:
    result = optimize_trajectory(problem)
    serialized_clearance = [
        None if np.isinf(item) else float(item) for item in result.minimum_clearance
    ]
    inputs: dict[str, Any] = {
        "dt": problem.dt,
        "goal": problem.goal.tolist(),
        "lower": problem.lower.tolist(),
        "start": problem.start.tolist(),
        "steps": problem.steps,
        "upper": problem.upper.tolist(),
    }
    if problem.seeds is not None:
        inputs["seeds"] = problem.seeds.tolist()
    if problem.collision_model is not None:
        inputs["collision"] = _collision_dict(problem.collision_model)
    return {
        "description": description,
        "expected": {
            "endpoint_error": result.endpoint_error.tolist(),
            "minimum_clearance": serialized_clearance,
            "objective": result.objective.tolist(),
            "selected_seed": result.selected_seed,
            "status": list(result.status),
            "success": result.success.tolist(),
            "trajectories": result.trajectories.tolist(),
        },
        "format": "curobo-metal-trajectory-case",
        "inputs": inputs,
        "name": name,
        "options": {
            "collision_subdivisions": problem.collision_subdivisions,
            "collision_tolerance": problem.collision_tolerance,
            "endpoint_tolerance": problem.endpoint_tolerance,
            "finite_difference_step": problem.finite_difference_step,
            "gradient_tolerance": problem.gradient_tolerance,
            "max_iterations": problem.max_iterations,
            "weights": vars(problem.weights),
        },
        "robot": robot_dict,
        "version": 1,
    }


def build_cases() -> dict[str, dict[str, Any]]:
    planar, planar_dict = _robot("two_link_planar.json")
    panda, panda_dict = _robot("panda_serial.json")
    pi_limits = np.full(2, np.pi)
    base_weights = TrajectoryWeights(endpoint=1000.0, joint_limit=10.0, velocity=0.01,
                                     acceleration=0.05, jerk=0.001)
    collision = CollisionModel(
        local_spheres=np.array([[0.0, 0.0, 0.0, 0.12]]),
        link_indices=np.array([2]),
        cuboid_centers=np.array([[1.7, 0.1, 0.0]]),
        cuboid_rotations=np.eye(3)[None],
        cuboid_half_extents=np.array([[0.12, 0.12, 0.2]]),
        activation_distance=0.15,
        weight=40.0,
    )
    cases = {
        "two_link_obstacle_free.json": _case(
            "two_link_obstacle_free", planar_dict,
            TrajectoryProblem(planar, np.array([-0.6, 0.2]), np.array([0.8, -0.3]),
                              -pi_limits, pi_limits, 7, 0.2, weights=base_weights,
                              max_iterations=40),
            "Minimum-jerk initialized obstacle-free planar motion.",
        ),
        "two_link_obstacle_detour.json": _case(
            "two_link_obstacle_detour", planar_dict,
            TrajectoryProblem(planar, np.array([-0.8, 0.0]), np.array([0.8, 0.0]),
                              -pi_limits, pi_limits, 9, 0.2, weights=base_weights,
                              collision_model=collision, collision_subdivisions=2,
                              max_iterations=100),
            "Swept tool-sphere sampling forces a deterministic cuboid detour.",
        ),
        "panda_obstacle_free.json": _case(
            "panda_obstacle_free", panda_dict,
            TrajectoryProblem(panda, np.array([-0.2, 0.1, 0.0, -0.8, 0.0, 0.7, 0.2]),
                              np.array([0.3, -0.2, 0.2, -1.1, 0.3, 1.0, -0.1]),
                              np.full(7, -2.5), np.full(7, 2.5), 8, 0.15,
                              weights=base_weights, max_iterations=30),
            "Seven-DoF Panda-class obstacle-free reference motion.",
        ),
        "infeasible.json": _case(
            "infeasible", planar_dict,
            TrajectoryProblem(planar, np.array([0.0, 0.0]), np.array([1.2, 0.0]),
                              np.array([-0.5, -0.5]), np.array([0.5, 0.5]), 6, 0.2,
                              weights=base_weights, max_iterations=20),
            "Goal conflicts with an inclusive joint limit.",
        ),
        "limit_constrained.json": _case(
            "limit_constrained", planar_dict,
            TrajectoryProblem(
                planar, np.array([-0.5, -0.25]), np.array([0.5, 0.25]),
                np.array([-0.5, -0.25]), np.array([0.5, 0.25]), 6, 0.2,
                seeds=np.array([np.linspace([-1.0, -0.8], [1.0, 0.8], 6)]),
                weights=base_weights, max_iterations=40,
            ),
            "Out-of-range seed is projected while endpoints lie exactly on limits.",
        ),
    }
    return cases


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> None:
    cases = build_cases()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for filename, value in cases.items():
        (FIXTURES / filename).write_text(canonical(value), encoding="utf-8")
    evidence = {
        "cases": {
            value["name"]: {
                "endpoint_error": value["expected"]["endpoint_error"],
                "minimum_clearance": value["expected"]["minimum_clearance"],
                "objective": value["expected"]["objective"],
                "selected_seed": value["expected"]["selected_seed"],
                "status": value["expected"]["status"],
                "success": value["expected"]["success"],
            }
            for value in cases.values()
        },
        "format": "curobo-metal-trajectory-correctness",
        "generator": "tests/fixtures/trajectory/generate.py",
        "reference": "curobo_metal.reference.trajectory",
        "version": 1,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(canonical(evidence), encoding="utf-8")


if __name__ == "__main__":
    main()
