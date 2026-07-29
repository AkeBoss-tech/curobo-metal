from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    CollisionModel,
    IKProblem,
    SerialRobot,
    collision_cost,
    forward_kinematics,
    joint_limit_cost,
    load_case,
    load_ik_case,
    matrix_to_quaternion,
    pose_cost,
    pose_error,
    problem_from_dict,
    quaternion_to_matrix,
    robot_collision_cost,
    save_ik_case,
    smoothness_cost,
    solve_ik,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
IK_FIXTURES = FIXTURES / "ik"


def _planar_robot() -> SerialRobot:
    return SerialRobot.from_dict(load_case(FIXTURES / "two_link_planar.json")["robot"])


def _numeric_gradient(function, value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    step = 1e-6
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        output[index] = (function(plus) - function(minus)) / (2 * step)
    return output


def test_quaternion_convention_round_trip_sign_and_pose_error() -> None:
    quaternion = np.array([-0.5, 0.5, -0.5, 0.5])
    rotation = quaternion_to_matrix(quaternion)
    canonical = matrix_to_quaternion(rotation)
    assert canonical[0] > 0
    np.testing.assert_allclose(quaternion_to_matrix(canonical), rotation, atol=1e-15)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [1.0, -2.0, 3.0]
    np.testing.assert_allclose(
        pose_error(transform, [0.5, -1.5, 2.0], canonical),
        [0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
        atol=1e-15,
    )
    with pytest.raises(ValueError):
        quaternion_to_matrix(np.zeros(4))


def test_pose_limit_collision_and_smoothness_gradients() -> None:
    error = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
    weights = np.arange(1.0, 7.0)
    actual = pose_cost(error, weights)
    np.testing.assert_allclose(
        actual.gradient, _numeric_gradient(lambda x: pose_cost(x, weights).value, error),
        rtol=3e-6, atol=3e-8,
    )
    q = np.array([-1.2, 0.1, 1.4])
    limits = joint_limit_cost(q, np.full(3, -1.0), np.full(3, 1.0), margin=0.1, weight=2)
    np.testing.assert_allclose(
        limits.gradient,
        _numeric_gradient(
            lambda x: joint_limit_cost(
                x, np.full(3, -1.0), np.full(3, 1.0), margin=0.1, weight=2
            ).value,
            q,
        ),
        rtol=3e-6, atol=3e-8,
    )
    trajectory = np.array([[0.0, 0.2], [0.3, -0.1], [0.9, 0.4], [1.1, 0.8]])
    smooth = smoothness_cost(trajectory, velocity_weight=0.7, acceleration_weight=1.3)
    np.testing.assert_allclose(
        smooth.gradient,
        _numeric_gradient(
            lambda x: smoothness_cost(
                x, velocity_weight=0.7, acceleration_weight=1.3
            ).value,
            trajectory,
        ),
        rtol=3e-6, atol=3e-8,
    )
    clearance = np.array([-0.2, 0.1, 0.8])
    collision = collision_cost(clearance, activation_distance=0.4, weight=2.0)
    np.testing.assert_allclose(
        collision.gradient,
        _numeric_gradient(
            lambda x: collision_cost(x, activation_distance=0.4, weight=2.0).value,
            clearance,
        ),
        rtol=3e-6, atol=3e-8,
    )


def test_collision_composition_uses_existing_references() -> None:
    robot = _planar_robot()
    model = CollisionModel(
        local_spheres=np.array([[0.0, 0.0, 0.0, 0.1]]),
        link_indices=np.array([2]),
        cuboid_centers=np.array([[1.75, 0.0, 0.0]]),
        cuboid_rotations=np.eye(3)[None],
        cuboid_half_extents=np.array([[0.05, 0.05, 0.05]]),
        activation_distance=0.0,
        weight=2.0,
    )
    # At q=0 the tool sphere contains the cuboid center: clearance=-0.15.
    assert robot_collision_cost(robot, np.zeros(2), model) == pytest.approx(0.0225)
    assert robot_collision_cost(robot, np.array([np.pi / 2, 0.0]), model) == 0.0


@pytest.mark.parametrize(
    ("name", "expected_status"),
    [
        ("reachable.json", "success"),
        ("unreachable.json", "infeasible_or_stationary"),
        ("limit_constrained.json", "limit_constrained"),
        ("collision_constrained.json", "collision_constrained"),
    ],
)
def test_replay_cases(name: str, expected_status: str) -> None:
    path = IK_FIXTURES / name
    case = load_ik_case(path)
    assert path.read_text() == (
        json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    result = solve_ik(problem_from_dict(case))
    assert result.status[0] == expected_status
    assert bool(result.success[0]) is bool(case["expected"]["success"][0])
    np.testing.assert_allclose(
        result.position_error, case["expected"]["position_error"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.rotation_error, case["expected"]["rotation_error"], atol=1e-10, rtol=0
    )


def test_seed_batch_selection_and_wrapping() -> None:
    robot = _planar_robot()
    desired = np.array([0.6, -0.8])
    target = forward_kinematics(robot, desired).transforms[0, -1]
    problem = IKProblem(
        robot, target[:3, 3], matrix_to_quaternion(target[:3, :3]),
        np.array([[0.0, 0.0], [1.0 + 4 * np.pi, -1.0]]),
        np.full(2, -np.pi), np.full(2, np.pi), np.ones(6),
        position_tolerance=2e-5, rotation_tolerance=2e-5,
        max_iterations=200, wrap_revolute=True,
    )
    result = solve_ik(problem)
    assert result.success.tolist() == [True, True]
    assert result.selected_seed in (0, 1)
    assert np.all(result.solutions >= -np.pi) and np.all(result.solutions <= np.pi)


def test_panda_class_chain_is_suitable_for_oracle_use() -> None:
    case = load_case(FIXTURES / "panda_serial.json")
    robot = SerialRobot.from_dict(case["robot"])
    desired = np.asarray(case["inputs"]["q"][1], dtype=np.float64)
    target = forward_kinematics(robot, desired).transforms[0, -1]
    result = solve_ik(IKProblem(
        robot, target[:3, 3], matrix_to_quaternion(target[:3, :3]),
        np.array([desired + 0.03]), np.full(7, -3.0), np.full(7, 3.0),
        np.ones(6), position_tolerance=1e-3, rotation_tolerance=1e-3,
        max_iterations=300,
    ))
    assert result.success.tolist() == [True]


def test_replay_validation_and_no_backend_dependency(tmp_path: Path) -> None:
    case = load_ik_case(IK_FIXTURES / "reachable.json")
    output = tmp_path / "case.json"
    save_ik_case(output, case)
    assert output.read_bytes() == (IK_FIXTURES / "reachable.json").read_bytes()
    source_modules = {
        importlib.import_module("curobo_metal.reference.costs"),
        importlib.import_module("curobo_metal.reference.ik"),
    }
    for module in source_modules:
        source = Path(module.__file__).read_text()
        assert "import torch" not in source
        assert "import curobo" not in source
        assert "import warp" not in source


def test_empty_seed_batch_and_invalid_problem_inputs() -> None:
    robot = _planar_robot()
    common = dict(
        robot=robot,
        target_position=np.zeros(3),
        target_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        seeds=np.empty((0, 2)),
        lower=np.full(2, -np.pi),
        upper=np.full(2, np.pi),
        pose_weights=np.ones(6),
    )
    empty = solve_ik(IKProblem(**common))
    assert empty.solutions.shape == (0, 2)
    assert empty.selected_seed is None
    with pytest.raises(TypeError):
        solve_ik(IKProblem(**dict(common, seeds=np.zeros((1, 2), dtype=int))))
    with pytest.raises(ValueError):
        solve_ik(IKProblem(**dict(common, target_quaternion=np.zeros(4))))
    with pytest.raises(ValueError):
        solve_ik(IKProblem(**dict(common, lower=np.ones(2), upper=np.zeros(2))))
    with pytest.raises(ValueError):
        solve_ik(IKProblem(**dict(common, position_tolerance=-1.0)))
