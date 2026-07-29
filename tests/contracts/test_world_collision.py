from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.reference import (
    Mesh,
    VoxelGrid,
    load_world_collision_case,
    mesh_distance,
    query_esdf,
    sample_voxel_sdf,
    save_world_collision_case,
    sphere_world_collision,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "world_collision"
ARTIFACT = Path(__file__).parents[2] / "artifacts" / "correctness" / "world_collision_reference.json"


def _meshes() -> dict[str, Mesh]:
    raw = json.loads((FIXTURES / "meshes.json").read_text())
    return {
        name: Mesh(np.array(v["vertices"], float), np.array(v["faces"], int), v["watertight"])
        for name, v in raw.items()
    }


def _grid(translation=(0.0, 0.0, 0.0), rotation=np.eye(3), oob=99.0) -> VoxelGrid:
    raw = json.loads((FIXTURES / "analytic_grid.json").read_text())
    return VoxelGrid(
        np.array(raw["values"], float), raw["voxel_size"],
        np.array(translation, float), np.array(rotation, float), oob,
    )


def test_box_signed_values_gradients_surface_and_face_tie() -> None:
    box = _meshes()["box"]
    points = np.array([[2.0, 0.2, -0.3], [0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [2.0, 2.0, 0.0]])
    result = mesh_distance(points, [box], np.zeros((1, 1, 3)), np.eye(3)[None, None])
    np.testing.assert_allclose(result.reduced_distance[0], [1, -1, 0, np.sqrt(2)], atol=1e-15)
    np.testing.assert_allclose(result.reduced_gradient[0, 0], [1, 0, 0], atol=1e-15)
    assert np.linalg.norm(result.reduced_gradient[0, 1]) == pytest.approx(1)
    np.testing.assert_allclose(result.reduced_gradient[0, 3], [2**-0.5, 2**-0.5, 0], atol=1e-15)
    assert result.winning_face[0, 1, 0] == 0  # first exact face tie


def test_mesh_transform_batches_environments_unsigned_and_inactive() -> None:
    triangle = _meshes()["open_triangle"]
    angle = np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    points = np.array([[[0.25, 0.25, 1.0]], [[1.75, -0.75, 1.0]]])
    translations = np.array([[[0, 0, 0]], [[2, -1, 0]]], float)
    rotations = np.array([[np.eye(3)], [rotation]])
    result = mesh_distance(points, [triangle], translations, rotations, env_indices=[0, 1], signed=False)
    np.testing.assert_allclose(result.reduced_distance, [[1], [1]], atol=1e-15)
    np.testing.assert_allclose(result.reduced_gradient[:, 0], [[0, 0, 1], [0, 0, 1]], atol=1e-15)
    with pytest.raises(ValueError, match="watertight"):
        mesh_distance(points[:1], [triangle], translations[:1], rotations[:1], signed=True)
    inactive = mesh_distance(points[:1], [triangle], translations[:1], rotations[:1],
                             env_mesh_active=[[False]], signed=False)
    assert np.isinf(inactive.reduced_distance[0, 0])
    assert inactive.winning_mesh[0, 0] == -1


def test_mesh_gradient_matches_finite_difference_inside_and_outside() -> None:
    box = _meshes()["box"]
    transform = (np.array([[[0.3, -0.2, 0.4]]]), np.eye(3)[None, None])
    for point in (np.array([[1.7, 0.1, 0.2]]), np.array([[0.4, 0.0, 0.1]])):
        actual = mesh_distance(point, [box], *transform)
        numeric = np.zeros(3)
        step = 1e-6
        for axis in range(3):
            plus, minus = point.copy(), point.copy()
            plus[0, axis] += step
            minus[0, axis] -= step
            numeric[axis] = (
                mesh_distance(plus, [box], *transform).reduced_distance[0, 0]
                - mesh_distance(minus, [box], *transform).reduced_distance[0, 0]
            ) / (2 * step)
        np.testing.assert_allclose(actual.reduced_gradient[0, 0], numeric, rtol=3e-6, atol=3e-8)


def test_affine_grid_exact_values_gradients_and_center_boundary() -> None:
    grid = _grid()
    points = np.array([[0.1, -0.2, 0.3], [-0.75, -0.75, -0.75], [0.75, 0.75, 0.75], [0.751, 0, 0]])
    result = sample_voxel_sdf(points, [[grid]])
    np.testing.assert_allclose(result.values[0, :3, 0], [-0.45, -1.875, 1.875], atol=1e-15)
    np.testing.assert_allclose(result.gradients[0, :3, 0], [[1, 2, -0.5]] * 3, atol=1e-15)
    assert np.all(result.valid[0, :3, 0])
    assert not result.valid[0, 3, 0]
    assert result.values[0, 3, 0] == 99
    np.testing.assert_array_equal(result.gradients[0, 3, 0], 0)


def test_transformed_grid_and_finite_difference_gradient() -> None:
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    grid = _grid((2, -1, 0.5), rotation)
    local = np.array([0.1, -0.2, 0.3])
    point = (rotation @ local + grid.translation)[None]
    actual = sample_voxel_sdf(point, [[grid]])
    expected_gradient = rotation @ np.array([1, 2, -0.5])
    assert actual.values[0, 0, 0] == pytest.approx(-0.45, abs=1e-15)
    np.testing.assert_allclose(actual.gradients[0, 0, 0], expected_gradient, atol=1e-15)
    numeric = np.zeros(3)
    step = 1e-6
    for axis in range(3):
        plus, minus = point.copy(), point.copy()
        plus[0, axis] += step
        minus[0, axis] -= step
        numeric[axis] = (
            sample_voxel_sdf(plus, [[grid]]).values[0, 0, 0]
            - sample_voxel_sdf(minus, [[grid]]).values[0, 0, 0]
        ) / (2 * step)
    np.testing.assert_allclose(actual.gradients[0, 0, 0], numeric, rtol=3e-6, atol=3e-8)


def test_esdf_batched_environment_reduction_padding_oob_and_tie() -> None:
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    envs = [[_grid(), _grid(oob=50)], [_grid((2, -1, 0.5), rotation), _grid((2, -1, 0.5), rotation)]]
    local = np.array([0.1, -0.2, 0.3])
    points = np.array([[local], [rotation @ local + np.array([2, -1, 0.5])]])
    result = query_esdf(points, envs, env_indices=[0, 1], padding=0.1)
    np.testing.assert_allclose(result.distance, [[-0.55], [-0.55]], atol=1e-15)
    np.testing.assert_array_equal(result.winning_grid, [[0], [0]])
    disabled = query_esdf(points[:1], envs, env_indices=[0], grid_active=[[False, True], [True, True]])
    assert disabled.winning_grid[0, 0] == 1
    oob = query_esdf(np.array([[10.0, 0, 0]]), envs)
    assert not oob.valid[0, 0] and np.isinf(oob.distance[0, 0])


def test_sphere_activation_values_boundaries_and_gradient() -> None:
    spheres = np.array([[0, 0, 0, 0.2], [0, 0, 0, 0.2], [0, 0, 0, 0.2]], float)
    sdf = np.array([0.35, 0.25, 0.1])
    grad = np.tile([1.0, 0.0, 0.0], (3, 1))
    cost, gradient = sphere_world_collision(spheres, sdf, grad, activation_distance=0.1, padding=0.05, weight=2)
    np.testing.assert_allclose(cost, [0, 0.1, 0.4], atol=1e-15)
    np.testing.assert_allclose(gradient, [[0, 0, 0], [-2, 0, 0], [-2, 0, 0]], atol=1e-15)
    linear, _ = sphere_world_collision(spheres[:1], [0.1], grad[:1], padding=0.05)
    assert linear[0] == pytest.approx(0.15)


def test_invalid_inputs_empty_queries_and_noncontiguous_float32() -> None:
    mesh = _meshes()["box"]
    with pytest.raises(ValueError, match="orthonormal"):
        mesh_distance(np.zeros((1, 3)), [mesh], np.zeros((1, 1, 3)), np.zeros((1, 1, 3, 3)))
    grid = _grid()
    empty = sample_voxel_sdf(np.empty((0, 3)), [[grid]])
    assert empty.values.shape == (1, 0, 1)
    backing = np.zeros((2, 6), dtype=np.float32)
    backing[:, ::2] = [[0.1, -0.2, 0.3], [0.2, -0.1, 0.1]]
    result = sample_voxel_sdf(backing[:, ::2], [[grid]])
    assert result.values.dtype == np.float64


def test_canonical_replay_and_artifact() -> None:
    replay = load_world_collision_case(ARTIFACT)
    assert ARTIFACT.read_text() == json.dumps(
        replay, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    assert replay["upstream_revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    assert {case["operation"] for case in replay["cases"]} == {
        "mesh_distance", "sample_voxel_sdf", "query_esdf", "sphere_world_collision"
    }
    temporary = ARTIFACT.with_suffix(".tmp")
    try:
        save_world_collision_case(temporary, replay)
        assert temporary.read_bytes() == ARTIFACT.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)
