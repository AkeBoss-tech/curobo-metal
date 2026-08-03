"""Batched lifecycle coverage for dense portable block storage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.storage import BlockSparseTSDF, BlockSparseTSDFCfg
from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper


def _storage(*, device: str = "cpu") -> BlockSparseTSDF:
    return BlockSparseTSDF(BlockSparseTSDFCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, origin=torch.tensor((0.1, -0.2, 0.3)),
        block_size=1, environments=2, device=device,
    ))


def test_storage_keeps_batched_environment_diagnostics_and_reset_isolation():
    storage = _storage()
    storage.prepare_frame()
    storage.state.weight[0, 0, 0, 0] = 1.0
    storage.state.weight[1, 1, 1, 1] = 2.0

    assert storage.get_data(0).new_blocks.tolist() == [0]
    assert storage.get_data(1).new_blocks.tolist() == [7]
    assert storage.get_stats()["active_blocks"] == 2
    assert storage.get_stats()["dense_capacity_voxels"] == 16
    assert storage.get_stats(environment=1)["active_blocks"] == 1
    assert storage.get_stats(environment=1)["dense_capacity_voxels"] == 8

    storage.reset(torch.tensor([1], dtype=torch.int64))
    assert storage.state.weight[0, 0, 0, 0].item() == 1.0
    assert storage.state.weight[1].sum().item() == 0.0
    assert storage.get_stats(environment=0)["observed_voxels"] == 1
    assert storage.get_stats(environment=1)["observed_voxels"] == 0
    with pytest.raises(ValueError, match="env_indices"):
        storage.reset(torch.tensor([2], dtype=torch.int64))


def test_storage_checkpoint_and_dense_import_are_clone_owned_and_validate_dtype():
    storage = _storage()
    storage.state.weight[1, 1, 0, 1] = 3.0
    storage.state.tsdf[1, 1, 0, 1] = -0.25
    checkpoint = storage.state_dict()
    checkpoint["weight"][1, 1, 0, 1] = 9.0
    assert storage.state.weight[1, 1, 0, 1].item() == 3.0

    storage.reset()
    storage.load_state_dict(storage.state_dict())
    assert storage.state.weight.sum().item() == 0.0
    storage.load_state_dict(checkpoint)
    assert storage.state.weight[1, 1, 0, 1].item() == 9.0
    blocks = storage.export_blocks()
    blocks["weight"] = blocks["weight"].to(torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        storage.import_blocks(blocks)


def test_storage_native_adaptation_accepts_nonstorage_depth_settings_and_cache_rebuilds():
    config = BlockSparseTSDFCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, origin=torch.zeros(3), block_size=1, environments=2, device="cpu",
    )
    native = PerceptionMapper(PerceptionConfig(
        shape=(2, 2, 2), voxel_size=0.1, grid_center=(0.0, 0.0, 0.0),
        truncation_distance=0.04, max_weight=1000.0, environments=2, block_size=1,
        depth_min=0.2, depth_max=4.0,
    ), device="cpu")
    storage = BlockSparseTSDF.from_native(config, native)
    first = storage.get_data(0).block_coords
    assert storage._coords_cache is first
    storage.invalidate_cache()
    assert storage._coords_cache is None
    rebuilt = storage.get_data(0).block_coords
    assert rebuilt.shape == first.shape and rebuilt.device == first.device
    with pytest.raises(ValueError, match="geometry/batch"):
        BlockSparseTSDF.from_native(
            BlockSparseTSDFCfg(grid_shape=(3, 2, 2), voxel_size=0.1, origin=torch.zeros(3), block_size=1, device="cpu"),
            native,
        )


@pytest.mark.parametrize("environments", [0, True])
def test_storage_configuration_rejects_invalid_batch_counts(environments):
    with pytest.raises(ValueError, match="environments"):
        BlockSparseTSDFCfg(grid_shape=(2, 2, 2), voxel_size=0.1, environments=environments)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_storage_batched_checkpoint_and_diagnostics_execute_on_mps_without_fallback():
    storage = _storage(device="mps")
    storage.prepare_frame()
    storage.state.weight[0, 0, 1, 0] = 1.0
    storage.state.weight[1, 1, 0, 1] = 1.0
    checkpoint = storage.state_dict()
    storage.reset(torch.tensor([0], dtype=torch.int64, device="mps"))
    storage.load_state_dict(checkpoint)
    assert storage.get_data(1).block_coords.device.type == "mps"
    assert storage.get_stats()["observed_voxels"] == 2
