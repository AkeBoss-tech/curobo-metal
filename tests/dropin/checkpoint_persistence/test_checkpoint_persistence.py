"""Portable mapper compact block checkpoint behavior."""

import pytest
import torch

from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    save_block_checkpoint,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _mapper(center=(0.0, 0.0, 0.2)):
    return Mapper(MapperCfg(
        (0.4, 0.4, 0.4), voxel_size=0.1, block_size=2,
        grid_center=torch.tensor(center), device="cpu",
    ))


def _observation():
    return CameraObservation(
        depth_image=torch.full((4, 4), 0.3),
        rgb_image=torch.full((4, 4, 3), 128, dtype=torch.uint8),
        intrinsics=torch.tensor(((10.0, 0.0, 1.5), (0.0, 10.0, 1.5), (0.0, 0.0, 1.0))),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]),
        depth_to_meter=1.0,
    )


def test_sparse_checkpoint_restores_cpu_state_and_checks_target_geometry(tmp_path):
    source = _mapper()
    source.integrate(_observation())
    expected = source._portable_sparse.export_blocks()
    path = tmp_path / "sparse-map.pt"
    source.save_blocks(path)

    target = _mapper()
    assert target.import_blocks(path) == len(expected["active_block_coords"])
    restored = target._portable_sparse.export_blocks()
    for name in expected:
        assert torch.equal(restored[name], expected[name])
    assert target._mapper.state.weight.sum() > 0
    with pytest.raises(ValueError, match="empty target"):
        target.import_blocks(path)
    with pytest.raises(ValueError, match="grid_center"):
        _mapper((0.0, 0.0, 0.0)).import_blocks(path)
    reweighted = _mapper()
    assert reweighted.import_blocks(path, import_weight=2.0) > 0
    active_weight = reweighted._portable_sparse.data.block_data[..., 1]
    assert active_weight[active_weight > 0].eq(2.0).all()
    with pytest.raises(ValueError, match="minimum_tsdf_weight"):
        _mapper().import_blocks(path, import_weight=0.01)


def test_external_sparse_checkpoint_is_validated_and_decoded(tmp_path):
    mapper = _mapper()
    metadata = build_block_metadata(mapper)
    size = metadata["block_size"] ** 3
    block_data = torch.zeros((1, size, 2), dtype=torch.float16)
    block_data[..., 0] = -0.25
    block_data[..., 1] = 1.0
    sparse = {
        "active_block_coords": torch.tensor([[0, 0, 0]], dtype=torch.int32),
        "block_data": block_data,
        "block_grid_rgb": torch.zeros((1, 1, 4), dtype=torch.float16),
    }
    path = tmp_path / "upstream-shape.pt"
    save_block_checkpoint(path, metadata, sparse)
    assert mapper.import_blocks(path) == 1
    assert int(mapper.tsdf.data.num_allocated.item()) == 1
    assert mapper._mapper.state.weight.sum() > 0
