"""Portable block-sparse ESDF lifecycle and sampling coverage."""

import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)


def _integrator(*, device: str = "cpu", blend: bool = False):
    return BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
        grid_shape=(4, 4, 4), esdf_grid_shape=(4, 4, 4), voxel_size=0.1,
        origin=torch.tensor((-0.2, -0.2, -0.2)), enable_static=True,
        blend_esdf=blend, dtype=torch.float32, device=device,
    ))


def _surface_scene():
    return SceneCfg(cuboid=[Cuboid(
        "surface", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )])


def test_esdf_window_blending_query_and_cache_lifecycle_are_real():
    plain, blended = _integrator(), _integrator(blend=True)
    plain.update_static_obstacles(_surface_scene())
    blended.update_static_obstacles(_surface_scene())
    assert not blended.is_esdf_current
    plain_field = plain.compute_esdf()
    blended_field = blended.compute_esdf()
    assert blended.is_esdf_current
    assert not torch.equal(plain_field, blended_field)

    origin = blended.origin
    # An exact one-cell shift is representable without inventing an arbitrary
    # resampling convention; moved-out cells deliberately become empty.
    shifted = blended.compute_esdf(origin + torch.tensor((0.1, 0.0, 0.0)))
    assert shifted.shape == (4, 4, 4)
    assert (blended._site_index >= 0).any()
    with pytest.raises(NotImplementedError, match="integer-voxel"):
        blended.compute_esdf(origin + torch.tensor((0.05, 0.0, 0.0)))

    sample = blended.query(torch.tensor([[0.0, 0.0, 0.0]]))
    assert sample.valid.shape == sample.distance.shape == (1, 1)
    assert sample.valid.item()
    assert blended.get_stats()["esdf_compute_count"] >= 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_esdf_window_and_query_stay_on_mps_without_fallback():
    integrator = _integrator(device="mps", blend=True)
    integrator.update_static_obstacles(_surface_scene())
    field = integrator.compute_esdf(integrator.origin + torch.tensor((0.1, 0.0, 0.0), device="mps"))
    result = integrator.query(torch.tensor([[0.0, 0.0, 0.0]], device="mps"))
    assert field.device.type == result.distance.device.type == result.gradient.device.type == "mps"
