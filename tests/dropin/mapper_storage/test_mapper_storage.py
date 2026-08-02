"""Behavioral checks for the dense block-storage compatibility adapter."""

import pytest
import torch

from curobo._src.geom.types import Capsule, Cuboid, SceneCfg, Sphere
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import BlockSparseTSDF, BlockSparseTSDFCfg


def test_dense_storage_lifecycle_exports_real_tensors_and_rejects_warp_abi():
    storage = BlockSparseTSDF(BlockSparseTSDFCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, origin=torch.tensor((0.2, 0.0, 0.1)), device="cpu",
    ))
    state = storage.state
    state.weight[0, 0, 0, 0] = 3.0
    state.tsdf[0, 0, 0, 0] = -0.5
    blocks = storage.export_blocks()
    storage.reset()
    storage.import_blocks(blocks)
    assert storage.get_stats()["observed_voxels"] == 1
    stats = storage.get_stats(scan_hash=True)
    assert stats["active_blocks"] == 1
    assert stats["num_allocated"] == 1
    assert stats["free_count"] == 0
    assert stats["holes"] == 0
    assert stats["hash_occ"] == 0
    assert stats["storage"] == "dense_portable"
    assert storage.data.block_data.shape == (1, 8, 2)
    assert storage.memory_usage_bytes() > 0
    with pytest.raises(NotImplementedError, match="Warp"):
        storage.get_warp_data()
    with pytest.raises(NotImplementedError, match="block-pool"):
        storage.import_blocks({"active_block_coords": torch.empty((0, 3), dtype=torch.int32)})


def test_prepare_frame_reports_new_dense_coverage_and_cache_reset_is_explicit():
    storage = BlockSparseTSDF(BlockSparseTSDFCfg(
        grid_shape=(4, 4, 4), voxel_size=0.1, block_size=2, origin=torch.zeros(3), device="cpu",
    ))
    storage.prepare_frame()
    storage.state.weight[0, 0, 0, 0] = 1.0
    storage.state.weight[0, 3, 3, 3] = 1.0
    data = storage.data
    assert torch.equal(data.new_blocks.cpu(), torch.tensor([0, 63], dtype=torch.int32))
    assert data.new_block_count.item() == 2
    stats = storage.get_stats()
    assert stats["active_blocks"] == 2
    assert stats["dense_logical_blocks"] == 8
    assert stats["pool_usage_pct"] == pytest.approx(25.0)

    storage.invalidate_cache()
    assert storage.data.new_block_count.item() == 0
    storage.reset()
    assert storage.get_stats()["active_blocks"] == 0
    assert storage.data.frustum_flags.sum().item() == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_dense_storage_frame_diagnostics_run_on_mps_without_cpu_fallback():
    storage = BlockSparseTSDF(BlockSparseTSDFCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, block_size=1, origin=torch.zeros(3), device="mps",
    ))
    storage.prepare_frame()
    storage.state.weight[0, 1, 0, 1] = 1.0
    assert storage.data.new_blocks.device.type == "mps"
    assert storage.get_stats()["active_blocks"] == 1


def test_occupied_voxel_queries_subvoxel_sampling_and_static_scene_replacement():
    mapper = Mapper(MapperCfg(
        (0.4, 0.4, 0.4), voxel_size=0.1, grid_center=torch.tensor((0.0, 0.0, 0.0)),
        enable_static=True, device="cpu",
    ))
    scene = SceneCfg(cuboid=[Cuboid("box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])])
    stamped = mapper.update_static_obstacles(scene)
    assert stamped > 0
    voxels = mapper.extract_occupied_voxels(surface_only=False, subvoxel_factor=2)
    assert len(voxels) == stamped * 8
    assert voxels.colors_uint8().shape == (len(voxels), 3)
    assert torch.equal(voxels.colors_uint8(), torch.full((len(voxels), 3), 128, dtype=torch.uint8))

    replacement = SceneCfg(sphere=[Sphere("sphere", position=[0.15, 0.15, 0.15], radius=0.01)])
    mapper.update_static_obstacles(replacement)
    assert mapper.get_stats()["static_voxels"] == 1
    with pytest.raises(NotImplementedError, match="mesh/voxel/Warp"):
        mapper.update_static_obstacles(SceneCfg(capsule=[Capsule("raw", base=[0, 0, 0], tip=[0, 0, .1], radius=.01)]))


def test_static_scene_requires_opt_in_and_esdf_window_boundary_is_explicit():
    mapper = Mapper(MapperCfg((0.2, 0.2, 0.2), voxel_size=0.1, device="cpu"))
    with pytest.raises(RuntimeError, match="enable_static"):
        mapper.update_static_obstacles(SceneCfg())
    with pytest.raises(NotImplementedError, match="sliding ESDF"):
        mapper.compute_esdf(esdf_origin=torch.tensor((1.0, 0.0, 0.0)))
