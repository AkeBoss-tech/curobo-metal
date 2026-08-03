"""Closure coverage for the portable ESDF integration lifecycle."""

import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)


def _integrator(*, device: str = "cpu") -> BlockSparseESDFIntegrator:
    return BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
        grid_shape=(4, 4, 4),
        esdf_grid_shape=(4, 4, 4),
        voxel_size=0.1,
        origin=torch.tensor((-0.2, -0.2, -0.2)),
        enable_static=True,
        dtype=torch.float32,
        device=device,
    ))


def _scene() -> SceneCfg:
    return SceneCfg(cuboid=[Cuboid(
        "surface", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )])


def test_cache_refresh_and_reset_have_truthful_lifecycle_state():
    integrator = _integrator()
    integrator.update_static_obstacles(_scene())
    first = integrator.compute_esdf()
    assert integrator.compute_esdf() is first
    assert integrator.get_stats()["esdf_compute_count"] == 1

    # A mutation invalidates the field.  A public grid request refreshes it,
    # rather than exposing the zeroed invalidation buffer as current data.
    integrator.clear_region((-0.2, -0.2, -0.2), (-0.1, -0.1, -0.1))
    assert not integrator.is_esdf_current
    grid = integrator.get_voxel_grid()
    assert integrator.is_esdf_current
    assert grid.feature_tensor is integrator.dist_field
    assert integrator.get_stats()["esdf_compute_count"] == 2

    integrator.reset()
    assert not integrator.is_esdf_current
    assert integrator.get_stats()["esdf_compute_count"] == 0
    assert torch.count_nonzero(integrator.dist_field) == 0
    assert torch.equal(integrator._site_index, torch.full_like(integrator._site_index, -1))


def test_rejected_fractional_window_preserves_prior_field_and_registration():
    integrator = _integrator()
    integrator.update_static_obstacles(_scene())
    integrator.compute_esdf()
    before_field = integrator.dist_field.clone()
    before_sites = integrator._site_index.clone()
    before_origin = integrator._last_esdf_origin.clone()
    before_generation = integrator._last_compute_generation.clone()
    before_count = integrator._compute_count

    with pytest.raises(NotImplementedError, match="integer-voxel"):
        integrator.compute_esdf(integrator.origin + torch.tensor((0.05, 0.0, 0.0)))

    assert torch.equal(integrator.dist_field, before_field)
    assert torch.equal(integrator._site_index, before_sites)
    assert torch.equal(integrator._last_esdf_origin, before_origin)
    assert torch.equal(integrator._last_compute_generation, before_generation)
    assert integrator._compute_count == before_count
    assert integrator.is_esdf_current


def test_esdf_input_boundaries_reject_cross_device_and_unsupported_resampling():
    integrator = _integrator()
    integrator.update_static_obstacles(_scene())
    with pytest.raises(NotImplementedError, match="resampling"):
        integrator.compute_esdf(esdf_voxel_size=torch.tensor([0.2]))
    if torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="share the ESDF field device"):
            integrator.compute_esdf(esdf_origin=integrator.origin.to("mps"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_public_refresh_and_device_validation_are_fallback_free():
    integrator = _integrator(device="mps")
    integrator.update_static_obstacles(_scene())
    grid = integrator.get_voxel_grid()
    assert integrator.is_esdf_current
    assert grid.feature_tensor.device.type == "mps"
    result = integrator.query(torch.zeros((1, 3), device="mps"))
    assert result.distance.device.type == result.gradient.device.type == "mps"
    with pytest.raises(ValueError, match="share the ESDF field device"):
        integrator.query(torch.zeros((1, 3)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_rejects_float64_storage_at_construction():
    with pytest.raises(TypeError, match="float16 or float32"):
        BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
            grid_shape=(2, 2, 2), esdf_grid_shape=(2, 2, 2),
            origin=torch.zeros(3), dtype=torch.float64, device="mps",
        ))
