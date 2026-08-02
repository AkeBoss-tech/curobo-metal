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
    assert storage.data.block_data.shape == (1, 8, 2)
    assert storage.memory_usage_bytes() > 0
    with pytest.raises(NotImplementedError, match="Warp"):
        storage.get_warp_data()
    with pytest.raises(NotImplementedError, match="block-pool"):
        storage.import_blocks({"active_block_coords": torch.empty((0, 3), dtype=torch.int32)})


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
