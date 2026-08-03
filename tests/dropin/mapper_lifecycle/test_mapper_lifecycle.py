"""Regression coverage for the portable high-level mapper lifecycle."""

import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _cfg(device="cpu", *, enable_static=False):
    return MapperCfg(
        extent_meters_xyz=(0.3, 0.4, 0.5), voxel_size=0.1,
        grid_center=torch.tensor((0.2, -0.1, 0.5)),
        truncation_distance=0.15, enable_static=enable_static,
        image_height=4, image_width=4, device=device,
    )


def _camera(device="cpu"):
    return CameraObservation(
        depth_image=torch.ones((4, 4), device=device),
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5),
                                 (0.0, 0.0, 1.0)), device=device),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device),
        depth_to_meter=1.0,
    )


def test_mapper_cfg_source_axis_order_round_trip_and_validation():
    cfg = _cfg()
    assert cfg.grid_shape == (5, 4, 3)
    assert cfg.native_grid_shape == (3, 4, 5)
    assert cfg.get_actual_extent() == pytest.approx((0.3, 0.4, 0.5))
    point = cfg.voxel_to_world(4, 3, 2)
    assert cfg.world_to_voxel(*point) == (4, 3, 2)
    assert cfg.world_to_voxel(100.0, 0.0, 0.0) == (-1, -1, -1)
    lo, hi = cfg.get_grid_bounds()
    assert lo == pytest.approx((0.05, -0.3, 0.25))
    assert hi == pytest.approx((0.35, 0.1, 0.75))

    with pytest.raises(ValueError, match="block_size"):
        MapperCfg((1, 1, 1), block_size=3)
    with pytest.raises(ValueError, match="depth_minimum"):
        MapperCfg((1, 1, 1), depth_minimum_distance=2.0, depth_maximum_distance=1.0)
    with pytest.raises(ValueError, match="image_height"):
        MapperCfg((1, 1, 1), image_height=4)
    with pytest.raises(ValueError, match="grid_center"):
        MapperCfg((1, 1, 1), grid_center=(0.0, 0.0))


def test_mapper_keeps_native_storage_order_and_static_layer_through_dynamic_clear():
    mapper = Mapper(_cfg(enable_static=True))
    assert mapper._mapper.state.tsdf.shape == (1, 3, 4, 5)
    scene = SceneCfg(cuboid=[Cuboid(
        "static", pose=[0.2, -0.1, 0.5, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )])
    assert mapper.update_static_obstacles(scene) > 0
    static_before = mapper._static_mask.clone()
    # The source clear operation is dynamic-only; it must not erase a static
    # scene channel merely because the requested AABB covers it.
    assert mapper.clear_region((-1, -1, -1), (1, 1, 1)) == 0
    assert torch.equal(mapper._static_mask, static_before)
    assert bool(mapper._mapper.state.occupancy[static_before].all().item())

    mapper.integrate(_camera())
    assert bool(mapper._mapper.state.occupancy[static_before].all().item())
    grid = mapper.compute_esdf()
    assert grid.feature_tensor.shape == (3, 4, 5)
    assert mapper.get_stats()["frame_count"] == 1
    mapper.reset()
    assert mapper.get_stats()["frame_count"] == mapper.get_stats()["esdf_compute_count"] == 0
    assert not bool(mapper._static_mask.any().item())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mapper_asymmetric_config_and_static_lifecycle_stay_on_mps():
    mapper = Mapper(_cfg("mps", enable_static=True))
    mapper.update_static_obstacles(SceneCfg(cuboid=[Cuboid(
        "static", pose=[0.2, -0.1, 0.5, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )]))
    mapper.integrate(_camera("mps"))
    grid = mapper.compute_esdf()
    assert mapper._mapper.state.tsdf.device.type == grid.feature_tensor.device.type == "mps"
