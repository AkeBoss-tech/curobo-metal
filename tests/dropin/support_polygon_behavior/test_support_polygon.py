"""Portable support-polygon behavior covered against the pinned V2 surface."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.cost_support_polygon import CostSupportPolygon
from curobo._src.cost.cost_support_polygon_cfg import CostSupportPolygonCfg
from curobo._src.types.device_cfg import DeviceCfg


def _feet(batch: int = 2, horizon: int = 3, *, device: str = "cpu") -> torch.Tensor:
    spheres = torch.zeros((batch, horizon, 5, 4), device=device)
    # A unit square in deliberately non-cyclic order verifies hull formation,
    # rather than accidentally relying on caller vertex ordering.
    spheres[:, :, 0, :2] = torch.tensor([1.0, 1.0], device=device)
    spheres[:, :, 1, :2] = torch.tensor([0.0, 0.0], device=device)
    spheres[:, :, 2, :2] = torch.tensor([1.0, 0.0], device=device)
    spheres[:, :, 3, :2] = torch.tensor([0.0, 1.0], device=device)
    spheres[:, :, :, 3] = 0.04
    return spheres


def _cost(*, device: str = "cpu", inside_cost_weight: float = 0.001) -> CostSupportPolygon:
    config = CostSupportPolygonCfg(
        weight=2.0,
        foot_sphere_indices=torch.tensor([0, 1, 2, 3]),
        inside_cost_weight=inside_cost_weight,
        device_cfg=DeviceCfg(torch.device(device)),
    )
    return CostSupportPolygon(config)


def test_batched_cost_uses_cached_contact_hulls_and_is_differentiable() -> None:
    cost = _cost()
    feet = _feet()
    com = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
            [[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 1.5, 0.0]],
        ],
        requires_grad=True,
    )

    result = cost(com, feet)
    assert result.shape == (2, 3)
    # An outside point pays its signed distance multiplied by the configured
    # scalar weight; an interior point pays only the small margin term.
    assert result[0, 1] > 0.7
    assert result[0, 0] < 0.01
    assert cost.vertices is not None and cost.vertices.shape[0] == 2
    result.sum().backward()
    assert com.grad is not None and torch.isfinite(com.grad).all()
    assert com.grad[0, 1, 0] > 0


def test_external_hull_supports_no_indices_and_rebuilds_on_batch_change() -> None:
    config = CostSupportPolygonCfg(weight=1.0, foot_sphere_indices=None)
    cost = CostSupportPolygon(config)
    vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    cached = cost.build_convex_hull(vertices)
    assert cached.shape[0] == 1
    com = torch.tensor([[[0.1, 0.1, 0.0]]])
    spheres = torch.zeros((1, 1, 1, 4))
    assert cost(com, spheres).shape == (1, 1)

    # A manually supplied single hull cannot silently be applied to a larger
    # batch.  This avoids an incorrect balance cost in a later batched solve.
    with pytest.raises(ValueError, match="incompatible"):
        cost(torch.zeros((2, 1, 3)), torch.zeros((2, 1, 1, 4)))


@pytest.mark.parametrize(
    ("com_shape", "sphere_shape", "match"),
    [
        ((1, 3), (1, 1, 4), "robot_com"),
        ((1, 1, 3), (1, 1, 4), "robot_spheres"),
        ((1, 2, 3), (1, 1, 5, 4), "batch/horizon"),
    ],
)
def test_shape_errors_are_explicit(com_shape, sphere_shape, match) -> None:
    cost = _cost()
    with pytest.raises(ValueError, match=match):
        cost(torch.zeros(com_shape), torch.zeros(sphere_shape))


def test_index_validation_and_empty_batches_are_well_defined() -> None:
    invalid = CostSupportPolygon(
        CostSupportPolygonCfg(weight=1.0, foot_sphere_indices=torch.tensor([7]))
    )
    with pytest.raises(ValueError, match="out-of-range"):
        invalid(torch.zeros((1, 1, 3)), torch.zeros((1, 1, 2, 4)))

    result = _cost()(torch.zeros((0, 2, 3)), torch.zeros((0, 2, 5, 4)))
    assert result.shape == (0, 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_support_polygon_stays_on_mps_without_cpu_fallback() -> None:
    cost = _cost(device="mps", inside_cost_weight=0.0)
    feet = _feet(batch=1, horizon=1, device="mps")
    com = torch.tensor([[[1.5, 0.5, 0.0]]], device="mps", requires_grad=True)
    result = cost(com, feet)
    result.sum().backward()
    assert result.device.type == com.grad.device.type == "mps"
    assert result.item() > 0
