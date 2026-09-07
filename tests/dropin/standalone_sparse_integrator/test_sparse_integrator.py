"""Regression coverage for the standalone portable sparse integrator path."""

import torch

from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    save_block_checkpoint,
)
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _observation(*, features: bool = False) -> CameraObservation:
    pose = Pose(
        position=torch.zeros(3),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    intrinsics = torch.tensor(
        [[20.0, 0.0, 3.5], [0.0, 20.0, 3.5], [0.0, 0.0, 1.0]]
    )
    kwargs = {}
    if features:
        kwargs["feature_grid"] = torch.ones((1, 1, 1, 2), dtype=torch.float16)
    return CameraObservation(
        depth_image=torch.full((8, 8), 0.2),
        rgb_image=torch.full((8, 8, 3), 100, dtype=torch.uint8),
        pose=pose,
        intrinsics=intrinsics,
        **kwargs,
    )


def _config() -> BlockSparseTSDFIntegratorCfg:
    return BlockSparseTSDFIntegratorCfg(
        voxel_size=0.01,
        origin=torch.zeros(3),
        grid_shape=(64, 64, 64),
        max_blocks=64,
        device="cpu",
        image_height=8,
        image_width=8,
        depth_minimum_distance=0.1,
        depth_maximum_distance=0.5,
        truncation_distance=0.05,
        feature_dim=2,
        feature_grid_height=1,
        feature_grid_width=1,
    )


def test_sparse_standalone_extracts_surface_occupied_and_features():
    integrator = BlockSparseTSDFIntegrator(_config())
    assert integrator.mapper is None
    integrator.integrate(_observation(features=True))

    occupied = integrator.extract_occupied_voxels()
    centers, colors, distances = integrator.extract_surface_voxels()

    assert len(occupied) > 0
    assert centers.shape[0] == colors.shape[0] == distances.shape[0]
    assert colors.dtype == torch.uint8
    assert torch.isfinite(distances).all()
    torch.testing.assert_close(occupied.features().mean(0), torch.ones(2), atol=1e-3, rtol=0)


def test_sparse_standalone_imports_compact_payload_and_checkpoint(tmp_path):
    source = BlockSparseTSDFIntegrator(_config())
    source.integrate(_observation(features=True))
    payload = source.tsdf.export_blocks()

    direct = BlockSparseTSDFIntegrator(_config())
    assert direct.import_blocks(payload) == len(payload["active_block_coords"])
    assert len(direct.extract_occupied_voxels()) == len(source.extract_occupied_voxels())

    path = tmp_path / "blocks.pt"
    save_block_checkpoint(path, build_block_metadata(source.tsdf), payload)
    restored = BlockSparseTSDFIntegrator(_config())
    assert restored.import_blocks(path) == len(payload["active_block_coords"])
    assert len(restored.extract_surface_voxels()[0]) == len(source.extract_surface_voxels()[0])


def test_sparse_esdf_delegates_extraction_and_computes_field():
    config = BlockSparseESDFIntegratorCfg(
        voxel_size=0.01,
        origin=torch.zeros(3),
        grid_shape=(64, 64, 64),
        esdf_grid_shape=(64, 64, 64),
        max_blocks=64,
        device="cpu",
        dtype=torch.float32,
        image_height=8,
        image_width=8,
        depth_minimum_distance=0.1,
        depth_maximum_distance=0.5,
        truncation_distance=0.05,
        feature_dim=2,
        feature_grid_height=1,
        feature_grid_width=1,
    )
    integrator = BlockSparseESDFIntegrator(config)
    assert integrator._tsdf_integrator.mapper is None
    integrator.integrate(_observation(features=True))
    assert len(integrator.extract_occupied_voxels()) > 0
    field = integrator.compute_esdf()
    assert field.shape == (64, 64, 64)
    assert torch.isfinite(field).all()
    assert integrator.is_esdf_current
