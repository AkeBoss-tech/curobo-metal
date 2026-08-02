"""Pinned V2 geometry collision/sphere-fit boundary coverage."""

from __future__ import annotations

import inspect

import pytest
import torch

from curobo._src.geom.collision import wp_autograd, wp_collision_common
from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from curobo._src.geom.collision.wp_speed_metric import apply_speed_metric
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    SWEEP_STEPS,
    swept_sphere_obstacle_collision_kernel,
)
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery, WarpSphereSDFFunction


class _Tetrahedron:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    is_watertight = True


def _names(callable_object):
    return list(inspect.signature(callable_object).parameters)


def test_raw_warp_boundaries_keep_pinned_launch_layouts() -> None:
    expected = [
        "obs_set", "spheres", "weight", "activation_distance", "env_query_idx",
        "distance", "gradient", "batch_size", "horizon", "num_spheres",
        "max_n_obs", "use_multi_env",
    ]
    assert _names(sphere_obstacle_collision_kernel) == expected
    assert _names(swept_sphere_obstacle_collision_kernel) == expected
    assert SWEEP_STEPS == 3
    assert _names(wp_autograd.SphereObstacleCollision.forward) == [
        "ctx", "query_spheres", "buffer", "scene", "weight", "activation_distance",
        "max_distance", "env_query_idx", "use_multi_env", "return_loss",
    ]
    assert _names(wp_autograd.SweptSphereObstacleCollision.forward) == [
        "ctx", "query_spheres", "buffer", "scene", "weight", "activation_distance",
        "max_distance", "speed_dt", "enable_speed_metric", "env_query_idx",
        "use_multi_env", "return_loss",
    ]
    with pytest.raises(NotImplementedError, match="Warp"):
        sphere_obstacle_collision_kernel(*([None] * len(expected)))
    with pytest.raises(NotImplementedError, match="Warp"):
        swept_sphere_obstacle_collision_kernel(*([None] * len(expected)))
    with pytest.raises(NotImplementedError, match="Warp"):
        wp_autograd.SphereObstacleCollision.apply()


def test_portable_collision_common_preserves_value_layout_and_accumulation() -> None:
    activation = wp_collision_common.apply_collision_activation(torch.tensor([0.0, 0.1, 0.3]), 0.2)
    torch.testing.assert_close(activation, torch.tensor([[0.0, 0.0], [0.025, 0.5], [0.2, 1.0]]))
    spheres = torch.tensor([[[[1.0, 2.0, 3.0, 0.4]]]])
    query = wp_collision_common.load_sphere_query(spheres, 0, 0.1)
    torch.testing.assert_close(query.center, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(query.radius_adjusted, torch.tensor(0.5))
    distance = torch.zeros(1, 1, 1)
    gradient = torch.zeros(1, 1, 1, 4)
    wp_collision_common.process_collision_result(
        torch.tensor([0.2, 1.0, 0.0, 0.0]), 0.5, 2.0, 0.2, 0, distance, gradient
    )
    torch.testing.assert_close(distance, torch.tensor([[[0.4]]]))
    torch.testing.assert_close(gradient[..., :3], torch.tensor([[[[2.0, 0.0, 0.0]]]]))


def test_speed_metric_has_pinned_inplace_layout_and_legacy_portable_layout() -> None:
    assert _names(apply_speed_metric) == [
        "spheres", "distance", "gradient", "speed_dt", "batch_size", "horizon", "num_spheres"
    ]
    spheres = torch.zeros(1, 4, 1, 4)
    spheres[0, :, 0, 0] = torch.tensor([0.0, 1.0, 3.0, 6.0])
    distance = torch.ones(1, 4, 1)
    gradient = torch.ones(1, 4, 1, 4)
    assert apply_speed_metric(spheres, distance, gradient, torch.tensor(1.0), 1, 4, 1) is None
    torch.testing.assert_close(distance[0, 1:3, 0], torch.tensor([1.5, 2.5]))
    # Kept for callers written against the first portable implementation.
    legacy_distance, legacy_gradient = apply_speed_metric(
        torch.ones(1, 4, 1), torch.ones(1, 4, 1, 4), spheres, torch.tensor(1.0)
    )
    torch.testing.assert_close(legacy_distance[0, 1:3, 0], torch.tensor([1.5, 2.5]))
    assert legacy_gradient.shape == (1, 4, 1, 4)


def test_portable_mesh_query_and_sdf_vjp_are_tensor_backed() -> None:
    query = WarpMeshQuery(_Tetrahedron(), torch.device("cpu"))
    points = torch.tensor([[2.0, 0.0, 0.0], [0.1, 0.1, 0.1]], requires_grad=True)
    distance, gradient = query.query_sdf(points)
    assert distance.shape == (2,)
    assert gradient.shape == (2, 3)
    assert query.query_outside_mask(points).tolist() == [True, False]
    closest, closest_distance = query.query_closest_point(points)
    assert closest.shape == points.shape
    torch.testing.assert_close(closest_distance, distance)
    WarpSphereSDFFunction.apply(points, query).sum().backward()
    assert points.grad is not None
