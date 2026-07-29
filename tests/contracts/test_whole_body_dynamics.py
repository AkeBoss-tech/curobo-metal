from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    TreeRobot,
    bias_torque,
    dynamics_cost,
    gravity_torque,
    inverse_dynamics,
    inverse_dynamics_derivatives,
    mass_matrix,
    tree_forward_kinematics,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "whole_body"
ARTIFACT = ROOT / "artifacts" / "correctness" / "whole_body_reference.json"


def load(name: str):
    path = FIXTURES / name
    case = json.loads(path.read_text())
    assert case["format"] == "curobo-metal-whole-body-case"
    assert case["version"] == 1
    return path, case, TreeRobot.from_dict(case["robot"])


def evaluated(name: str):
    _, case, robot = load(name)
    q = np.asarray(case["inputs"]["q"], dtype=np.float64)
    qd = np.asarray(case["inputs"]["qd"], dtype=np.float64)
    qdd = np.asarray(case["inputs"]["qdd"], dtype=np.float64)
    return robot, q, qd, qdd


def test_tree_topology_multiple_effectors_and_sibling_jacobians() -> None:
    robot, q, _, _ = evaluated("branched_toy.json")
    result = tree_forward_kinematics(robot, q)
    assert result.transforms.shape == (2, 6, 4, 4)
    assert result.geometric_jacobian.shape == (2, 6, 6, 3)
    assert tuple(result.link_names[i] for i in result.end_effector_indices) == (
        "left_tool", "right_tool", "slider"
    )
    # The slider and arm branches do not influence one another.
    np.testing.assert_array_equal(result.geometric_jacobian[:, 4, :, :2], 0)
    np.testing.assert_array_equal(result.geometric_jacobian[:, 2:4, :, 2], 0)


def test_mimic_affine_semantics_and_derivative() -> None:
    robot, _, _, _ = evaluated("branched_toy.json")
    q = np.array([0.0, 0.4, 0.0])
    result = tree_forward_kinematics(robot, q)
    angle = -0.5 * q[1] + 0.1
    relative = np.linalg.inv(result.transforms[0, 1]) @ result.transforms[0, 3]
    np.testing.assert_allclose(
        relative[:3, :3],
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        atol=2e-15,
    )
    step = 1e-6
    plus = tree_forward_kinematics(robot, q + [0, step, 0]).transforms[0, 3]
    minus = tree_forward_kinematics(robot, q - [0, step, 0]).transforms[0, 3]
    np.testing.assert_allclose(
        result.transform_jacobian[0, 3, :, :, 1],
        (plus - minus) / (2 * step),
        rtol=2e-7, atol=2e-9,
    )


@pytest.mark.parametrize("name", ["branched_toy.json", "panda_inertial.json"])
def test_rnea_mass_bias_gravity_consistency(name: str) -> None:
    robot, q, qd, qdd = evaluated(name)
    torque = inverse_dynamics(robot, q, qd, qdd).torque
    matrix = mass_matrix(robot, q)
    bias = bias_torque(robot, q, qd)
    np.testing.assert_allclose(
        torque, np.einsum("bij,bj->bi", matrix, qdd) + bias, rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        gravity_torque(robot, q),
        inverse_dynamics(robot, q, np.zeros_like(q), np.zeros_like(q)).torque,
        rtol=0, atol=2e-12,
    )
    np.testing.assert_allclose(matrix, matrix.transpose(0, 2, 1), atol=2e-12)
    assert np.min(np.linalg.eigvalsh(matrix)) > 1e-5


def test_inverse_dynamics_finite_difference_acceleration_jacobian() -> None:
    robot, q, qd, qdd = evaluated("branched_toy.json")
    derivatives = inverse_dynamics_derivatives(robot, q, qd, qdd)
    np.testing.assert_allclose(
        derivatives.d_tau_d_qdd, mass_matrix(robot, q), rtol=2e-6, atol=2e-7
    )
    assert derivatives.d_tau_d_q.shape == (2, 3, 3)
    assert derivatives.d_tau_d_qd.shape == (2, 3, 3)
    assert np.all(np.isfinite(derivatives.d_tau_d_q))


def _potential(robot: TreeRobot, q: np.ndarray) -> float:
    transforms = tree_forward_kinematics(robot, q).transforms[0]
    total = 0.0
    for transform, link in zip(transforms, robot.links):
        position = transform[:3, :3] @ link.com + transform[:3, 3]
        total -= link.mass * float(np.dot(robot.gravity, position))
    return total


@pytest.mark.parametrize("name", ["branched_toy.json", "panda_inertial.json"])
def test_gravity_is_potential_energy_gradient(name: str) -> None:
    robot, q_batch, _, _ = evaluated(name)
    q = q_batch[1]
    analytical = gravity_torque(robot, q)[0]
    numeric = np.empty(robot.dof)
    step = 1e-6
    for j in range(robot.dof):
        delta = np.zeros(robot.dof)
        delta[j] = step
        numeric[j] = (_potential(robot, q + delta) - _potential(robot, q - delta)) / (2 * step)
    np.testing.assert_allclose(analytical, numeric, rtol=2e-6, atol=2e-7)


def test_batched_unbatched_strided_empty_and_validation() -> None:
    robot, q, qd, qdd = evaluated("branched_toy.json")
    backing = np.zeros(6, dtype=np.float32)
    backing[::2] = q[1]
    strided = backing[::2]
    assert not strided.flags.c_contiguous
    single = inverse_dynamics(robot, strided, qd[1].astype(np.float32), qdd[1].astype(np.float32))
    batch = inverse_dynamics(
        robot, strided[None], qd[1:2].astype(np.float32), qdd[1:2].astype(np.float32)
    )
    assert single.input_was_batched is False
    np.testing.assert_array_equal(single.torque, batch.torque)
    empty = tree_forward_kinematics(robot, np.empty((0, robot.dof)))
    assert empty.transforms.shape == (0, len(robot.links), 4, 4)
    with pytest.raises(TypeError):
        tree_forward_kinematics(robot, [0, 0, 0])
    with pytest.raises(ValueError):
        inverse_dynamics(robot, q, qd[:1], qdd)
    with pytest.raises(ValueError):
        tree_forward_kinematics(robot, np.full(robot.dof, np.nan))


def test_torque_limits_and_costs() -> None:
    robot, _, _, _ = evaluated("branched_toy.json")
    torque = np.array([[6.0, -10.0, 5.0], [13.0, 0.0, -7.0]])
    cost = dynamics_cost(robot, torque, effort_weight=0.5, limit_weight=3.0)
    np.testing.assert_array_equal(cost.effort, 0.5 * np.sum(torque**2, axis=1))
    np.testing.assert_array_equal(cost.limit, [12.0, 15.0])
    np.testing.assert_array_equal(cost.total, cost.effort + cost.limit)


def test_checked_in_reference_artifact_replays() -> None:
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["format"] == "curobo-metal-whole-body-reference"
    assert artifact["version"] == 1
    for name, expected in artifact["cases"].items():
        path, case, robot = load(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected["fixture_sha256"]
        q = np.asarray(case["inputs"]["q"], dtype=np.float64)
        qd = np.asarray(case["inputs"]["qd"], dtype=np.float64)
        qdd = np.asarray(case["inputs"]["qdd"], dtype=np.float64)
        fk = tree_forward_kinematics(robot, q)
        np.testing.assert_allclose(fk.transforms, expected["transforms"], atol=2e-11, rtol=0)
        np.testing.assert_allclose(
            fk.geometric_jacobian, expected["geometric_jacobian"], atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(
            inverse_dynamics(robot, q, qd, qdd).torque,
            expected["inverse_dynamics"], atol=2e-10, rtol=0,
        )
        np.testing.assert_allclose(mass_matrix(robot, q), expected["mass_matrix"], atol=5e-10)
        np.testing.assert_allclose(gravity_torque(robot, q), expected["gravity"], atol=2e-10)
        np.testing.assert_allclose(bias_torque(robot, q, qd), expected["bias"], atol=2e-10)
        derivatives = inverse_dynamics_derivatives(robot, q, qd, qdd)
        np.testing.assert_allclose(
            derivatives.d_tau_d_q, expected["d_tau_d_q"], atol=2e-7, rtol=2e-6
        )
        np.testing.assert_allclose(
            derivatives.d_tau_d_qd, expected["d_tau_d_qd"], atol=2e-7, rtol=2e-6
        )
        np.testing.assert_allclose(
            derivatives.d_tau_d_qdd, expected["d_tau_d_qdd"], atol=2e-7, rtol=2e-6
        )


def test_invalid_topology_and_mimic_source_are_rejected() -> None:
    _, case, _ = load("branched_toy.json")
    broken = json.loads(json.dumps(case["robot"]))
    broken["links"][2]["parent"] = 4
    with pytest.raises(ValueError, match="topological"):
        TreeRobot.from_dict(broken)
    broken = json.loads(json.dumps(case["robot"]))
    broken["links"][3]["joint"]["mimic"]["joint"] = "missing"
    with pytest.raises(ValueError, match="mimic source"):
        TreeRobot.from_dict(broken)
