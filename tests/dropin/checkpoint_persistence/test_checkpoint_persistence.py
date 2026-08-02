"""Portable mapper checkpoint behavior and explicit sparse-pool boundary."""

import pytest
import torch

from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    save_block_checkpoint,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg


def _mapper(center=(0.2, -0.1, 0.3)):
    return Mapper(MapperCfg(
        (0.2, 0.2, 0.2), voxel_size=0.1,
        grid_center=torch.tensor(center), device="cpu",
    ))


def test_dense_checkpoint_restores_cpu_state_and_checks_target_geometry(tmp_path):
    source = _mapper()
    state = source._mapper.state
    state.tsdf[0, 0, 0, 0] = -0.25
    state.weight[0, 0, 0, 0] = 3.0
    state.occupancy[0, 0, 0, 0] = True
    state.esdf[0, 0, 0, 0] = -0.1
    state.gradient[0, 0, 0, 0] = torch.tensor((1.0, 0.0, 0.0))
    state.generation[0] = 7
    path = tmp_path / "dense-map.pt"
    source.save_blocks(path)

    target = _mapper()
    assert target.import_blocks(path) == 1
    restored = target._mapper.state
    for name in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation"):
        assert torch.equal(getattr(restored, name), getattr(state, name))
    with pytest.raises(ValueError, match="empty target"):
        target.import_blocks(path)
    with pytest.raises(ValueError, match="grid_center"):
        _mapper((0.0, 0.0, 0.0)).import_blocks(path)
    reweighted = _mapper()
    assert reweighted.import_blocks(path, import_weight=2.0) == 1
    assert reweighted._mapper.state.weight[0, 0, 0, 0].item() == 2.0
    with pytest.raises(ValueError, match="minimum_tsdf_weight"):
        _mapper().import_blocks(path, import_weight=0.01)


def test_sparse_checkpoint_is_validated_but_never_decoded_as_dense(tmp_path):
    mapper = _mapper()
    metadata = build_block_metadata(mapper)
    size = metadata["block_size"] ** 3
    sparse = {
        "active_block_coords": torch.tensor([[0, 0, 0]], dtype=torch.int32),
        "block_data": torch.zeros((1, size, 2), dtype=torch.float16),
        "block_grid_rgb": torch.zeros((1, 1, 4), dtype=torch.float16),
    }
    path = tmp_path / "upstream-shape.pt"
    save_block_checkpoint(path, metadata, sparse)
    with pytest.raises(NotImplementedError, match="CUDA/Warp sparse block-pool"):
        mapper.import_blocks(path)
