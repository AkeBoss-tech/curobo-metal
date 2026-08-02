from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    CollisionModel,
    SerialRobot,
    TrajectoryProblem,
    TrajectoryWeights,
    TRAJECTORY_REPLAY_ATOL,
    endpoint_cost,
    interpolated_states,
    load_case,
    load_trajectory_case,
    minimum_jerk_trajectory,
    optimize_trajectory,
    save_trajectory_case,
    smoothness_cost,
    trajectory_collision_cost,
    trajectory_problem_from_dict,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TRAJECTORY_FIXTURES = FIXTURES / "trajectory"


def _numeric_gradient(function, value: np.ndarray, step: float = 1e-6) -> np.ndarray:
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2.0 * step)
    return gradient


def _planar() -> SerialRobot:
    return SerialRobot.from_dict(load_case(FIXTURES / "two_link_planar.json")["robot"])


def test_minimum_jerk_definition_and_endpoint_cost_gradient() -> None:
    start, goal = np.array([-1.0, 0.5]), np.array([2.0, -0.5])
    actual = minimum_jerk_trajectory(start, goal, 5)
    phase = np.linspace(0.0, 1.0, 5)
    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    np.testing.assert_allclose(actual, start + blend[:, None] * (goal - start), atol=1e-15)
    np.testing.assert_array_equal(actual[[0, -1]], [start, goal])
    perturbed = actual.copy()
    perturbed[[0, -1]] += [[0.2, -0.1], [-0.3, 0.4]]
    cost = endpoint_cost(perturbed, start, goal, weight=2.3)
    numeric = _numeric_gradient(lambda q: endpoint_cost(q, start, goal, weight=2.3).value,
                                perturbed)
    np.testing.assert_allclose(cost.gradient, numeric, rtol=3e-6, atol=3e-8)


def test_velocity_acceleration_jerk_dt_gradient_and_scaling() -> None:
    q = np.array([[0.0, 0.2], [0.1, -0.3], [0.5, 0.4], [1.2, 0.1], [1.4, 0.8]])
    kwargs = dict(velocity_weight=0.7, acceleration_weight=1.1, jerk_weight=0.3, dt=0.2)
    cost = smoothness_cost(q, **kwargs)
    np.testing.assert_allclose(
        cost.gradient,
        _numeric_gradient(lambda value: smoothness_cost(value, **kwargs).value, q),
        rtol=3e-6, atol=3e-7,
    )
    velocity_only = smoothness_cost(q, velocity_weight=1.0, dt=0.2).value
    assert velocity_only == pytest.approx(0.5 / 0.2 * np.sum(np.diff(q, axis=0) ** 2))


def test_interpolation_coordinates_samples_and_noncontiguous_input() -> None:
    backing = np.zeros((3, 4))
    backing[:, ::2] = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    q = backing[:, ::2]
    assert not q.flags.c_contiguous
    states, coordinates = interpolated_states(q, 2)
    np.testing.assert_array_equal(coordinates, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_array_equal(
        states, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    )


def test_swept_collision_detects_between_knot_collision_and_gradient() -> None:
    case = load_trajectory_case(TRAJECTORY_FIXTURES / "two_link_obstacle_detour.json")
    problem = trajectory_problem_from_dict(case)
    straight = minimum_jerk_trajectory(problem.start, problem.goal, problem.steps)
    discrete, discrete_clearance, _ = trajectory_collision_cost(
        problem.robot, straight[::2], problem.collision_model, subdivisions=1
    )
    swept, swept_clearance, _ = trajectory_collision_cost(
        problem.robot, straight[::2], problem.collision_model, subdivisions=2
    )
    assert np.min(swept_clearance) <= np.min(discrete_clearance)
    assert swept.value >= discrete.value
    # Check the composed interpolation/FK/collision finite-difference gradient
    # on a compact three-knot slice away from box medial boundaries.
    q = np.array([[-0.8, 0.0], [-0.07, -0.03], [0.8, 0.0]])
    actual = trajectory_collision_cost(
        problem.robot, q, problem.collision_model, subdivisions=2
    )[0]
    numeric = _numeric_gradient(
        lambda value: trajectory_collision_cost(
            problem.robot, value, problem.collision_model, subdivisions=2
        )[0].value,
        q,
    )
    np.testing.assert_allclose(actual.gradient, numeric, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    ("name", "status", "success"),
    [
        ("two_link_obstacle_free.json", "success", True),
        ("two_link_obstacle_detour.json", "success", True),
        ("panda_obstacle_free.json", "success", True),
        ("infeasible.json", "endpoint_infeasible", False),
        ("limit_constrained.json", "success", True),
    ],
)
def test_canonical_replay_and_oracle_result(name: str, status: str, success: bool) -> None:
    path = TRAJECTORY_FIXTURES / name
    case = load_trajectory_case(path)
    assert path.read_text() == json.dumps(
        case, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    result = optimize_trajectory(trajectory_problem_from_dict(case))
    assert result.status == (status,)
    assert result.success.tolist() == [success]
    assert result.selected_seed == case["expected"]["selected_seed"]
    # The canonical fixture records a 1e-10 output grid.  This permits the
    # last-bit libm/BLAS variation of the finite-difference oracle without
    # accepting a numerically meaningful trajectory or objective regression.
    np.testing.assert_allclose(result.trajectories, case["expected"]["trajectories"],
                               rtol=0.0, atol=TRAJECTORY_REPLAY_ATOL)
    np.testing.assert_allclose(result.objective, case["expected"]["objective"],
                               rtol=0.0, atol=TRAJECTORY_REPLAY_ATOL)
    np.testing.assert_array_equal(result.trajectories, np.round(result.trajectories, 10))
    np.testing.assert_array_equal(result.objective, np.round(result.objective, 10))
    if name == "two_link_obstacle_detour.json":
        assert result.minimum_clearance[0] > 0.0
        assert np.max(np.abs(result.trajectories[0, 1:-1, 1])) > 0.02
    if name == "limit_constrained.json":
        problem = trajectory_problem_from_dict(case)
        assert np.all(result.trajectories >= problem.lower)
        assert np.all(result.trajectories <= problem.upper)


def test_seed_batch_is_independent_and_first_exact_tie_wins() -> None:
    robot = _planar()
    start, goal = np.array([-0.2, 0.1]), np.array([0.3, -0.1])
    seed = minimum_jerk_trajectory(start, goal, 5)
    problem = TrajectoryProblem(
        robot, start, goal, np.full(2, -1.0), np.full(2, 1.0), 5, 0.1,
        seeds=np.stack((seed, seed)), max_iterations=5,
        weights=TrajectoryWeights(velocity=0.0, acceleration=0.0, jerk=0.0),
    )
    result = optimize_trajectory(problem)
    assert result.success.tolist() == [True, True]
    assert result.selected_seed == 0
    np.testing.assert_array_equal(result.trajectories[0], result.trajectories[1])
    empty = optimize_trajectory(TrajectoryProblem(
        robot, start, goal, np.full(2, -1.0), np.full(2, 1.0), 5, 0.1,
        seeds=np.empty((0, 5, 2)), max_iterations=5,
    ))
    assert empty.trajectories.shape == (0, 5, 2)
    assert empty.status == ()
    assert empty.selected_seed is None


def test_fixture_and_artifact_regeneration_is_byte_deterministic() -> None:
    location = TRAJECTORY_FIXTURES / "generate.py"
    spec = importlib.util.spec_from_file_location("trajectory_fixture_generator", location)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.build_cases()
    for filename, value in generated.items():
        assert (TRAJECTORY_FIXTURES / filename).read_text() == module.canonical(value)
    artifact = json.loads((ROOT / "artifacts/correctness/trajectory_reference.json").read_text())
    assert artifact["format"] == "curobo-metal-trajectory-correctness"
    assert set(artifact["cases"]) == {value["name"] for value in generated.values()}


def test_serialization_validation_invalid_inputs_and_no_backend_dependency(tmp_path: Path) -> None:
    case = load_trajectory_case(TRAJECTORY_FIXTURES / "two_link_obstacle_free.json")
    target = tmp_path / "trajectory.json"
    save_trajectory_case(target, case)
    assert target.read_bytes() == (TRAJECTORY_FIXTURES / "two_link_obstacle_free.json").read_bytes()
    with pytest.raises(ValueError, match="unsupported"):
        save_trajectory_case(target, dict(case, version=2))
    with pytest.raises(ValueError):
        minimum_jerk_trajectory(np.zeros(2), np.ones(2), 1)
    with pytest.raises(TypeError):
        minimum_jerk_trajectory(np.zeros(2, dtype=int), np.ones(2), 3)
    problem = trajectory_problem_from_dict(case)
    with pytest.raises(ValueError):
        optimize_trajectory(TrajectoryProblem(**{**vars(problem), "dt": 0.0}))
    with pytest.raises(ValueError):
        optimize_trajectory(TrajectoryProblem(**{**vars(problem), "seeds": np.zeros((1, 2, 2))}))
    for name in ("trajectory", "costs"):
        source = Path(importlib.import_module(f"curobo_metal.reference.{name}").__file__).read_text()
        for forbidden in ("import torch", "import curobo", "import warp", "import isaac"):
            assert forbidden not in source.lower()
