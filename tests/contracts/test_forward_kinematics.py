from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import SerialRobot, forward_kinematics, load_case, save_case

FIXTURES = Path(__file__).parents[1] / "fixtures"


def evaluated_fixture(name: str):
    case = load_case(FIXTURES / name)
    robot = SerialRobot.from_dict(case["robot"])
    return case, robot, forward_kinematics(robot, case["inputs"]["q"])


@pytest.mark.parametrize("name", ["two_link_planar.json", "panda_serial.json"])
def test_checked_in_golden_replay(name: str) -> None:
    case, _, actual = evaluated_fixture(name)
    expected = case["expected"]
    assert actual.link_names == tuple(expected["link_names"])
    np.testing.assert_allclose(actual.transforms, expected["transforms"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        actual.transform_jacobian, expected["transform_jacobian"], rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(
        actual.geometric_jacobian, expected["geometric_jacobian"], rtol=0, atol=1e-13
    )


def test_two_link_has_hand_derived_pose_and_jacobian() -> None:
    _, robot, result = evaluated_fixture("two_link_planar.json")
    # q=[0, 0]: tool origin is at 1 + 0.75 on x.
    np.testing.assert_array_equal(result.transforms[0, -1], np.array(
        [[1, 0, 0, 1.75], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    ))
    # Linear columns are z x (p_tool - p_joint); angular columns are z.
    np.testing.assert_array_equal(
        result.geometric_jacobian[0, -1],
        np.array(
            [[0, 0], [1.75, 0.75], [0, 0], [0, 0], [0, 0], [1, 1]],
            dtype=np.float64,
        ),
    )
    # q=[pi/2, 0] rotates the entire arm onto +y.
    np.testing.assert_allclose(result.transforms[1, -1, :3, 3], [0, 1.75, 0], atol=2e-16)
    assert robot.dof == 2


@pytest.mark.parametrize("fixture", ["two_link_planar.json", "panda_serial.json"])
def test_transform_jacobian_matches_central_difference(fixture: str) -> None:
    case, robot, actual = evaluated_fixture(fixture)
    q = np.asarray(case["inputs"]["q"], dtype=np.float64)
    step = 1e-6
    for batch_index in range(q.shape[0]):
        for column in range(robot.dof):
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[batch_index, column] += step
            q_minus[batch_index, column] -= step
            plus = forward_kinematics(robot, q_plus).transforms[batch_index]
            minus = forward_kinematics(robot, q_minus).transforms[batch_index]
            numeric = (plus - minus) / (2 * step)
            np.testing.assert_allclose(
                actual.transform_jacobian[batch_index, :, :, :, column],
                numeric,
                rtol=2e-7,
                atol=2e-9,
            )


def test_batched_unbatched_float32_and_non_contiguous_inputs() -> None:
    case = load_case(FIXTURES / "panda_serial.json")
    robot = SerialRobot.from_dict(case["robot"])
    backing = np.zeros((14,), dtype=np.float32)
    backing[::2] = np.asarray(case["inputs"]["q"][1], dtype=np.float32)
    q = backing[::2]
    assert not q.flags.c_contiguous
    unbatched = forward_kinematics(robot, q)
    batched = forward_kinematics(robot, q[None, :])
    assert unbatched.input_was_batched is False
    assert batched.input_was_batched is True
    assert unbatched.transforms.shape == (1, 8, 4, 4)
    assert unbatched.transform_jacobian.shape == (1, 8, 4, 4, 7)
    assert unbatched.geometric_jacobian.shape == (1, 8, 6, 7)
    assert unbatched.transforms.dtype == np.float64
    np.testing.assert_array_equal(unbatched.transforms, batched.transforms)


def test_transforms_remain_rigid_at_large_wrapped_angles() -> None:
    case = load_case(FIXTURES / "panda_serial.json")
    robot = SerialRobot.from_dict(case["robot"])
    q = np.arange(robot.dof, dtype=np.float64) * (2000 * np.pi) + 0.25
    rotations = forward_kinematics(robot, q).transforms[0, :, :3, :3]
    identity = np.broadcast_to(np.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), identity, atol=2e-12)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=2e-12)


def test_vjp_matches_scalar_central_difference() -> None:
    case = load_case(FIXTURES / "panda_serial.json")
    robot = SerialRobot.from_dict(case["robot"])
    q = np.asarray(case["inputs"]["q"][1], dtype=np.float64)
    weights = np.arange(8 * 4 * 4, dtype=np.float64).reshape(8, 4, 4) / 127.0
    result = forward_kinematics(robot, q)
    analytical = np.einsum("lrc,lrcj->j", weights, result.transform_jacobian[0])
    step = 1e-6
    numeric = np.empty(robot.dof)
    for column in range(robot.dof):
        delta = np.zeros(robot.dof)
        delta[column] = step
        plus = np.sum(weights * forward_kinematics(robot, q + delta).transforms[0])
        minus = np.sum(weights * forward_kinematics(robot, q - delta).transforms[0])
        numeric[column] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytical, numeric, rtol=2e-7, atol=2e-9)


def test_empty_batch_and_prismatic_joint() -> None:
    robot = SerialRobot.from_dict(
        {
            "name": "slider",
            "joints": [
                {
                    "name": "slide",
                    "type": "prismatic",
                    "axis": [2.0, 0.0, 0.0],
                    "origin": {"xyz": [0.0, 1.0, 0.0]},
                }
            ],
        }
    )
    empty = forward_kinematics(robot, np.empty((0, 1), dtype=np.float64))
    assert empty.transforms.shape == (0, 1, 4, 4)
    result = forward_kinematics(robot, np.array([0.5], dtype=np.float64))
    np.testing.assert_array_equal(result.transforms[0, 0, :3, 3], [0.5, 1.0, 0.0])
    np.testing.assert_array_equal(
        result.geometric_jacobian[0, 0, :, 0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    ("q", "error"),
    [
        ([0.0], ValueError),
        ([[0.0, 0.0, 0.0]], ValueError),
        ([0, 0], TypeError),
        ([np.nan, 0.0], ValueError),
        ([np.inf, 0.0], ValueError),
    ],
)
def test_invalid_inputs_are_rejected(q, error) -> None:
    case = load_case(FIXTURES / "two_link_planar.json")
    robot = SerialRobot.from_dict(case["robot"])
    with pytest.raises(error):
        forward_kinematics(robot, q)


def test_replay_serialization_is_canonical_and_validated(tmp_path: Path) -> None:
    case = load_case(FIXTURES / "two_link_planar.json")
    destination = tmp_path / "case.json"
    save_case(destination, case)
    assert destination.read_text().endswith("\n")
    assert json.loads(destination.read_text()) == case
    assert destination.read_text() == (
        json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    broken = dict(case, version=999)
    destination.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="unsupported"):
        load_case(destination)
