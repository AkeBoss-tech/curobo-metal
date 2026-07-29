import numpy as np
import pytest

from curobo_metal.reference.perception import (
    CameraObservation, PerceptionConfig, dense_esdf, empty_state, integrate, voxel_centers,
)


def _camera(depth):
    return CameraObservation(
        np.asarray(depth, dtype=np.float64),
        np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
        np.eye(4),
    )


def test_voxel_centers_are_xyz_center_aligned():
    cfg = PerceptionConfig((3, 3, 3), 0.5, grid_center=(1.0, 2.0, 3.0))
    centers = voxel_centers(cfg)
    np.testing.assert_allclose(centers[0, 0, 0], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(centers[1, 1, 1], [1.0, 2.0, 3.0])


def test_invalid_depth_is_ignored_and_reset_state_is_empty():
    cfg = PerceptionConfig((3, 3, 3), 0.25, depth_min=0.1, depth_max=2.0)
    initial = empty_state(cfg)
    result = integrate(cfg, initial, _camera([[np.nan, 0, np.inf], [-1, 3, 0], [0, 0, 0]]))
    np.testing.assert_array_equal(result.weight, 0)
    np.testing.assert_array_equal(result.tsdf, 1)
    np.testing.assert_array_equal(result.esdf, cfg.unobserved_esdf)


def test_plane_depth_creates_signed_field_and_repeated_updates_accumulate():
    cfg = PerceptionConfig((3, 3, 5), 0.25, grid_center=(0, 0, 0.5),
                           truncation_distance=0.25, depth_min=0.1, depth_max=2)
    state = integrate(cfg, empty_state(cfg), _camera(np.ones((3, 3))))
    assert np.any(state.occupancy)
    assert np.all(state.esdf[state.occupancy] < 0)
    assert np.all(state.esdf[~state.occupancy] > 0)
    twice = integrate(cfg, state, _camera(np.ones((3, 3))))
    assert np.max(twice.weight) == 2
    np.testing.assert_allclose(twice.tsdf, state.tsdf)


def test_input_validation_is_explicit():
    with pytest.raises(ValueError, match="shape"):
        PerceptionConfig((1, 2, 3), 0.1)
    cfg = PerceptionConfig((2, 2, 2), 0.1)
    bad = CameraObservation(np.ones((2, 2)), np.eye(3), np.diag([1, 1, 1, 2]))
    with pytest.raises(ValueError, match="bottom row"):
        integrate(cfg, empty_state(cfg), bad)


def test_dense_esdf_distance_and_gradient_for_single_voxel():
    occupancy = np.zeros((3, 3, 3), dtype=bool)
    occupancy[1, 1, 1] = True
    esdf, gradient = dense_esdf(occupancy, 0.2, 1.0)
    assert esdf[1, 1, 1] == pytest.approx(-0.1)
    assert esdf[2, 1, 1] == pytest.approx(0.1)
    np.testing.assert_allclose(gradient[2, 1, 1], [1, 0, 0])
