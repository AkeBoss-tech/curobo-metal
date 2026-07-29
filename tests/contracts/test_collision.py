from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    load_collision_case,
    save_collision_case,
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
    transform_spheres,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "collision"


def test_sphere_transform_batched_unbatched_inactive_and_jacobian() -> None:
    transforms = np.stack((np.eye(4), np.eye(4)))
    transforms[1, :3, 3] = [1.0, 2.0, 3.0]
    local = np.array([[0.25, -0.5, 1.0, 0.2], [9.0, 9.0, 9.0, 0.3]])
    actual = transform_spheres(transforms, local, [1, 0], [True, False])
    assert not actual.input_was_batched
    np.testing.assert_array_equal(actual.spheres[0, 0], [1.25, 1.5, 4.0, 0.2])
    np.testing.assert_array_equal(actual.spheres[0, 1], 0.0)
    np.testing.assert_array_equal(
        actual.center_transform_jacobian[0, 0, 1, 1], [0.25, -0.5, 1.0, 1.0]
    )
    batched = transform_spheres(transforms[None], local, [1, 0], [True, False])
    assert batched.input_was_batched
    np.testing.assert_array_equal(actual.spheres, batched.spheres)


def test_transform_jacobian_matches_finite_difference() -> None:
    transform = np.array(
        [[[0.0, -1.0, 0.0, 0.4], [1.0, 0.0, 0.0, -0.2],
          [0.0, 0.0, 1.0, 0.7], [0.0, 0.0, 0.0, 1.0]]]
    )
    local = np.array([[0.2, -0.3, 0.5, 0.1]])
    result = transform_spheres(transform, local, [0])
    step = 1e-6
    for row in range(3):
        for column in range(4):
            plus, minus = transform.copy(), transform.copy()
            plus[0, row, column] += step
            minus[0, row, column] -= step
            numeric = (
                transform_spheres(plus, local, [0]).spheres[0, 0, row]
                - transform_spheres(minus, local, [0]).spheres[0, 0, row]
            ) / (2 * step)
            assert result.center_transform_jacobian[0, 0, row, row, column] == pytest.approx(
                numeric, abs=2e-10
            )


def test_sphere_pair_sign_contact_masks_padding_and_first_tie() -> None:
    spheres = np.array(
        [[0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5]]
    )
    result = sphere_sphere_signed_distance(
        spheres, [[0, 1], [0, 2], [1, 2]], pair_active=[True, True, False]
    )
    np.testing.assert_array_equal(result.distances[0, :2], [0.0, 0.0])
    assert np.isinf(result.distances[0, 2])
    assert result.winning_pair.tolist() == [0]
    np.testing.assert_array_equal(result.reduced_gradient[0, 0], [-1.0, 0.0, 0.0, -1.0])
    padded = sphere_sphere_signed_distance(spheres, [[0, 1]], padding=0.1)
    assert padded.reduced_distance[0] == pytest.approx(-0.1)
    inactive = sphere_sphere_signed_distance(
        spheres, [[0, 1]], sphere_active=[True, False, True]
    )
    assert np.isinf(inactive.reduced_distance[0])
    assert inactive.winning_pair[0] == -1
    np.testing.assert_array_equal(inactive.reduced_gradient, 0.0)


def test_coincident_sphere_subgradient_is_deterministic() -> None:
    spheres = np.array([[0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.3]])
    result = sphere_sphere_signed_distance(spheres, [[0, 1]])
    assert result.reduced_distance[0] == -0.5
    np.testing.assert_array_equal(result.gradients[0, 0, 0], [1.0, 0.0, 0.0, -1.0])
    np.testing.assert_array_equal(result.gradients[0, 0, 1], [-1.0, 0.0, 0.0, -1.0])


def test_sphere_pair_gradient_matches_finite_difference() -> None:
    spheres = np.array([[0.2, -0.1, 0.4, 0.15], [1.1, 0.3, -0.2, 0.25]])
    actual = sphere_sphere_signed_distance(spheres, [[0, 1]])
    step = 1e-6
    numeric = np.empty_like(spheres)
    for sphere in range(2):
        for component in range(4):
            plus, minus = spheres.copy(), spheres.copy()
            plus[sphere, component] += step
            minus[sphere, component] -= step
            numeric[sphere, component] = (
                sphere_sphere_signed_distance(plus, [[0, 1]]).reduced_distance[0]
                - sphere_sphere_signed_distance(minus, [[0, 1]]).reduced_distance[0]
            ) / (2 * step)
    np.testing.assert_allclose(actual.reduced_gradient[0], numeric, rtol=3e-6, atol=3e-8)


def test_box_values_inside_outside_contact_and_boundary_subgradient() -> None:
    box_args = (
        np.array([[0.0, 0.0, 0.0]]),
        np.array([np.eye(3)]),
        np.array([[1.0, 1.0, 1.0]]),
    )
    spheres = np.array(
        [[0.0, 0.0, 0.0, 0.25], [1.25, 0.0, 0.0, 0.25],
         [2.0, 2.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    )
    result = sphere_cuboid_signed_distance(spheres, *box_args)
    np.testing.assert_allclose(
        result.reduced_distance[0], [-1.25, 0.0, np.sqrt(2.0), 0.0], atol=1e-15
    )
    # Center ties x/y/z and edge ties x/y: both select lowest axis, sign(0)=+.
    np.testing.assert_array_equal(result.reduced_sphere_gradient[0, 0], [1, 0, 0, -1])
    np.testing.assert_array_equal(result.reduced_sphere_gradient[0, 3], [1, 0, 0, -1])
    np.testing.assert_allclose(
        result.reduced_sphere_gradient[0, 2, :3], [2**-0.5, 2**-0.5, 0], atol=1e-15
    )


def test_rotated_box_gradient_matches_finite_difference() -> None:
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    spheres = np.array([[1.2, 0.8, 0.6, 0.17]])
    args = (np.array([[0.1, -0.2, 0.05]]), rotation[None], np.array([[0.4, 0.3, 0.2]]))
    actual = sphere_cuboid_signed_distance(spheres, *args)
    step = 1e-6
    numeric = np.empty(4)
    for component in range(4):
        plus, minus = spheres.copy(), spheres.copy()
        plus[0, component] += step
        minus[0, component] -= step
        numeric[component] = (
            sphere_cuboid_signed_distance(plus, *args).reduced_distance[0, 0]
            - sphere_cuboid_signed_distance(minus, *args).reduced_distance[0, 0]
        ) / (2 * step)
    np.testing.assert_allclose(
        actual.reduced_sphere_gradient[0, 0], numeric, rtol=3e-6, atol=3e-8
    )


def test_empty_batches_pairs_cuboids_and_spheres() -> None:
    pair = sphere_sphere_signed_distance(np.empty((0, 2, 4)), np.empty((0, 2), dtype=int))
    assert pair.distances.shape == (0, 0)
    cuboid = sphere_cuboid_signed_distance(
        np.empty((0, 3, 4)), np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 3))
    )
    assert cuboid.distances.shape == (0, 3, 0)
    no_spheres = sphere_cuboid_signed_distance(
        np.empty((0, 4)), np.zeros((1, 3)), np.eye(3)[None], np.ones((1, 3))
    )
    assert no_spheres.reduced_distance.shape == (1, 0)


def test_noncontiguous_float32_input_and_output_dtype() -> None:
    backing = np.zeros((2, 8), dtype=np.float32)
    backing[:, ::2] = [[0, 0, 0, 0.2], [1, 0, 0, 0.2]]
    spheres = backing[:, ::2]
    assert not spheres.flags.c_contiguous
    result = sphere_sphere_signed_distance(spheres, [[0, 1]])
    assert result.distances.dtype == np.float64
    assert result.reduced_distance[0] == pytest.approx(0.6, abs=2e-8)


def test_batched_queries_preserve_independent_batch_rows() -> None:
    spheres = np.array(
        [
            [[0.0, 0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.0, 0.2], [0.3, 0.0, 0.0, 0.2]],
        ]
    )
    pairs = sphere_sphere_signed_distance(spheres, [[0, 1]])
    assert not pairs.input_was_unbatched
    np.testing.assert_allclose(pairs.reduced_distance, [0.6, -0.1], atol=1e-15)
    boxes = sphere_cuboid_signed_distance(
        spheres[:, :1], np.zeros((1, 3)), np.eye(3)[None], np.ones((1, 3))
    )
    assert boxes.reduced_distance.shape == (2, 1)
    np.testing.assert_array_equal(boxes.reduced_distance, [[-1.2], [-1.2]])


@pytest.mark.parametrize("name", ["minimal.json", "realistic.json"])
def test_checked_in_replay_cases(name: str) -> None:
    path = FIXTURES / name
    replay = load_collision_case(path)
    assert path.read_text() == (
        json.dumps(replay, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    for case in replay["cases"]:
        inputs, expected = case["inputs"], case["expected"]
        if case["operation"] == "transform_spheres":
            result = transform_spheres(
                np.asarray(inputs["transforms"], dtype=float),
                np.asarray(inputs["local_spheres"], dtype=float),
                inputs["link_indices"],
                inputs.get("active"),
            )
            np.testing.assert_allclose(
                result.spheres, expected["spheres"], **case["tolerance"]
            )
        elif case["operation"] == "sphere_sphere":
            result = sphere_sphere_signed_distance(
                np.asarray(inputs["spheres"], dtype=float),
                inputs["pairs"],
                sphere_active=inputs.get("sphere_active"),
                pair_active=inputs.get("pair_active"),
                padding=inputs.get("padding", 0.0),
            )
            np.testing.assert_allclose(
                result.distances[:, : len(expected["distance"][0])],
                expected["distance"], **case["tolerance"]
            )
            np.testing.assert_array_equal(result.winning_pair, expected["winner"])
            if "gradient" in expected:
                np.testing.assert_allclose(result.gradients, expected["gradient"], atol=1e-13)
        else:
            result = sphere_cuboid_signed_distance(
                np.asarray(inputs["spheres"], dtype=float),
                np.asarray(inputs["centers"], dtype=float),
                np.asarray(inputs["rotations"], dtype=float),
                np.asarray(inputs["half_extents"], dtype=float),
                cuboid_active=inputs.get("cuboid_active"),
                padding=inputs.get("padding", 0.0),
            )
            np.testing.assert_array_equal(result.winning_cuboid, expected["winner"])
            np.testing.assert_allclose(
                result.reduced_distance, expected["distance"], **case["tolerance"]
            )
            if "all_distances" in expected:
                np.testing.assert_allclose(
                    result.distances, expected["all_distances"], **case["tolerance"]
                )
            np.testing.assert_allclose(
                result.reduced_sphere_gradient, expected["gradient"], **case["tolerance"]
            )


def test_replay_serialization_is_canonical_and_rejects_nonfinite(tmp_path: Path) -> None:
    case = load_collision_case(FIXTURES / "minimal.json")
    target = tmp_path / "collision.json"
    save_collision_case(target, case)
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    assert target.read_text() == canonical
    with pytest.raises(ValueError, match="unsupported"):
        save_collision_case(target, dict(case, version=2))
    with pytest.raises(ValueError):
        save_collision_case(target, dict(case, extra=float("inf")))


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda: sphere_sphere_signed_distance([[0] * 4], []), TypeError),
        (lambda: sphere_sphere_signed_distance(np.zeros((1, 4)), [[0, 0]]), ValueError),
        (lambda: sphere_cuboid_signed_distance(
            np.zeros((1, 4)), np.zeros((1, 3)), np.zeros((1, 3, 3)), np.ones((1, 3))
        ), ValueError),
        (lambda: transform_spheres(np.eye(4)[None], np.array([[0., 0., 0., -1.]]), [0]), ValueError),
    ],
)
def test_invalid_inputs(call, error) -> None:
    with pytest.raises(error):
        call()
