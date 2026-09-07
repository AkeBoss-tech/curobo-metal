"""Large high-level maps select bounded sparse storage."""

import torch
import pytest

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS unavailable"
    )),
])
def test_large_sparse_extent_integrates_queries_and_clears_without_dense_allocation(device):
    config = MapperCfg(
        extent_meters_xyz=(10.0, 10.0, 10.0),
        voxel_size=0.02,
        block_size=8,
        device=device,
    )
    mapper = Mapper(config)
    assert mapper._sparse_only
    assert tuple(mapper._mapper.state.tsdf.shape) == (1, 2, 2, 2)
    mapper.integrate(CameraObservation(
        depth_image=torch.ones((4, 4), device=device),
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5),
                                 (0.0, 0.0, 1.0)), device=device),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device),
        depth_to_meter=1.0,
    ))
    stats = mapper.get_stats()
    assert stats["tsdf_storage"] == "block_sparse"
    assert stats["active_blocks"] > 0
    result = mapper.query(torch.tensor([[0.0, 0.0, 1.0]], device=device))
    assert result.distance.shape == result.valid.shape == (1, 1)
    assert result.valid.all()
    grid = mapper.compute_esdf()
    assert grid.feature_tensor.numel() < 1_000_000
    assert mapper.is_esdf_current
    depth = mapper.render_depth(
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5),
                      (0.0, 0.0, 1.0)), device=device),
        Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device),
        (4, 4),
    )
    assert depth.shape == (4, 4)
    assert mapper.clear_region((-0.5, -0.5, 0.5), (0.5, 0.5, 1.5)) > 0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS unavailable"
    )),
])
def test_large_sparse_extent_stamps_static_primitives_without_dense_mirror(device):
    mapper = Mapper(MapperCfg(
        extent_meters_xyz=(10.0, 10.0, 10.0), voxel_size=0.02,
        block_size=8, enable_static=True, device=device,
    ))
    count = mapper.update_static_obstacles(SceneCfg(cuboid=[Cuboid(
        "box", pose=[0, 0, 1, 1, 0, 0, 0], dims=[0.2, 0.2, 0.2],
    )]))
    assert count > 0
    assert mapper.get_stats()["static_voxels"] == count
    assert mapper.query(torch.tensor([[0.0, 0.0, 1.0]], device=device)).valid.all()
