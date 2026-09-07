"""Focused regressions for the portable high-level block mapper."""

from __future__ import annotations

import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _observation(*, center=(1.2, -0.7, 2.0), rgb=None, features=None):
    h, w = 8, 10
    depth = torch.full((h, w), float(center[2]))
    if rgb is None:
        rgb = torch.full((h, w, 3), 64, dtype=torch.uint8)
    return CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        feature_grid=features,
        intrinsics=torch.tensor(((40.0, 0.0, w / 2), (0.0, 40.0, h / 2), (0.0, 0.0, 1.0))),
        pose=Pose.from_list([center[0], center[1], 0, 1, 0, 0, 0]),
        depth_to_meter=1.0,
    )


def _cfg(**kwargs):
    values = dict(
        extent_meters_xyz=(0.5, 0.3, 0.4), voxel_size=0.1,
        grid_center=torch.tensor((1.2, -0.7, 2.0)), truncation_distance=0.15,
        image_height=8, image_width=10, block_size=2, device="cpu",
    )
    values.update(kwargs)
    return MapperCfg(**values)


def test_sparse_origin_and_checkpoint_roundtrip_preserve_world_positions(tmp_path):
    mapper = Mapper(_cfg())
    cfg = mapper.config
    assert torch.allclose(mapper.tsdf.data.origin, cfg.grid_center)
    mapper.integrate(_observation())
    before = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.15)
    assert len(before) > 0
    checkpoint = tmp_path / "nonzero-center.pt"
    mapper.save_blocks(checkpoint)

    restored = Mapper(_cfg())
    restored.import_blocks(checkpoint)
    after = restored.extract_occupied_voxels(surface_only=True, sdf_threshold=0.15)
    assert len(after) == len(before)
    torch.testing.assert_close(after.centers, before.centers)
    assert torch.allclose(after.centers[:, 2].median(), torch.tensor(2.0), atol=0.15)


def test_surface_extraction_excludes_saturated_free_space_and_coords_keep_logical_shape():
    """Surface voxels stay near the measured plane with a lazy block pool."""
    mapper = Mapper(_cfg())
    mapper.integrate(_observation())

    data = mapper.tsdf.data
    # Coordinate metadata retains the source-shaped max_blocks contract even
    # though the much larger voxel/appearance payloads grow on demand.
    assert data.block_coords.view(data.max_blocks, 3).shape == (data.max_blocks, 3)
    assert data.block_data.shape[0] < data.max_blocks

    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.15)
    assert len(surface) > 0
    # A saturated +1 TSDF is free space in front of the surface and must not
    # be returned merely because the threshold equals truncation_distance.
    assert torch.all((surface.centers[:, 2] - 2.0).abs() <= 0.1)


def test_tilted_surface_extraction_stays_on_analytic_plane():
    """Saturated free-space cells must not bias a tilted surface estimate."""
    height, width = 16, 20
    focal = 40.0
    intrinsics = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    normal = torch.tensor([0.25, -0.10, 1.0])
    normal = normal / normal.norm()
    point = torch.tensor([0.0, 0.0, 2.0])
    u = torch.arange(width).view(1, width).expand(height, width).float()
    v = torch.arange(height).view(height, 1).expand(height, width).float()
    rays = torch.stack(
        ((u - intrinsics[0, 2]) / focal, (v - intrinsics[1, 2]) / focal,
         torch.ones_like(u)), dim=-1
    )
    depth = (normal @ point) / (rays * normal).sum(dim=-1).clamp_min(1e-4)

    mapper = Mapper(_cfg(
        extent_meters_xyz=(0.5, 0.4, 0.4),
        grid_center=point,
        truncation_distance=0.1,
        image_height=height,
        image_width=width,
    ))
    mapper.integrate(CameraObservation(
        name="tilted", depth_image=depth, rgb_image=torch.full((height, width, 3), 127, dtype=torch.uint8),
        intrinsics=intrinsics, pose=Pose.from_list([0.0, 0.0, 0.0, 1, 0, 0, 0]),
        depth_to_meter=1.0,
    ))
    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.1)
    assert len(surface) > 0
    distance = torch.abs((surface.centers - point).matmul(normal))
    assert distance.median() <= 2.0 * mapper.config.voxel_size


def test_rgb_and_features_accumulate_and_are_exposed_by_extraction():
    mapper = Mapper(_cfg(feature_dim=2, feature_grid_height=2, feature_grid_width=2))
    rgb = torch.zeros((8, 10, 3), dtype=torch.uint8)
    rgb[..., 0] = 240
    feature_grid = torch.zeros((1, 2, 2, 2), dtype=torch.float16)
    feature_grid[..., 1] = 1
    mapper.integrate(_observation(rgb=rgb, features=feature_grid))
    data = mapper.tsdf.data
    n = int(data.num_allocated.item())
    assert n > 0
    assert bool((data.block_grid_rgb[:n, 0, 3] > 0).any().item())
    assert bool((data.block_feature_weight[:n] > 0).any().item())
    voxels = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.15)
    assert len(voxels) > 0
    assert bool((voxels.colors_uint8()[:, 0] > voxels.colors_uint8()[:, 1]).any().item())
    sampled_features = voxels.features()
    assert bool((sampled_features[:, 1] > 0.9).any().item())


def test_clear_deallocates_sparse_blocks_and_reuses_capacity():
    mapper = Mapper(_cfg())
    mapper.integrate(_observation())
    high_water = int(mapper.tsdf.data.num_allocated.item())
    assert high_water > 0
    assert mapper.clear_region(*mapper.config.get_grid_bounds()) > 0
    assert mapper.get_stats()["active_blocks"] == 0
    assert int(mapper.tsdf.data.free_count.item()) > 0
    mapper.integrate(_observation())
    assert mapper.get_stats()["active_blocks"] > 0
    assert int(mapper.tsdf.data.num_allocated.item()) == high_water


def test_static_only_state_is_persisted_and_restored(tmp_path):
    mapper = Mapper(_cfg(enable_static=True))
    scene = SceneCfg(cuboid=[Cuboid("static", pose=[1.2, -0.7, 2.0, 1, 0, 0, 0], dims=[0.2, 0.2, 0.2])])
    assert mapper.update_static_obstacles(scene) > 0
    assert int(mapper.tsdf.data.num_allocated.item()) > 0
    checkpoint = tmp_path / "static-only.pt"
    mapper.save_blocks(checkpoint)
    restored = Mapper(_cfg(enable_static=True))
    restored.import_blocks(checkpoint)
    assert restored.get_stats()["static_voxels"] > 0
    assert restored.extract_occupied_voxels(surface_only=False).__len__() > 0


def test_render_produces_normals_color_and_textured_mesh():
    mapper = Mapper(_cfg())
    rgb = torch.zeros((8, 10, 3), dtype=torch.uint8)
    rgb[..., 1] = 200
    rgb[..., 2] = 20
    observation = _observation(rgb=rgb)
    mapper.integrate(observation)
    intrinsics = observation.intrinsics
    depth, normals, valid = mapper.render(intrinsics, observation.pose, (8, 10))
    assert bool(valid.any().item())
    assert bool((normals.norm(dim=-1)[valid] > 0).any().item())
    _, _, color, _ = mapper.render_color(intrinsics, observation.pose, (8, 10))
    assert bool((color.sum(dim=-1) > 0).any().item())
    mesh = mapper.extract_textured_mesh(observation)
    assert mesh.vertex_colors is not None
    assert bool((torch.as_tensor(mesh.vertex_colors).sum(dim=-1) > 0).any().item())


def test_feature_matching_returns_ranked_matches():
    mapper = Mapper(_cfg(feature_dim=2, feature_grid_height=1, feature_grid_width=1))
    features = torch.zeros((1, 1, 1, 2), dtype=torch.float16)
    features[..., 0] = 1
    mapper.integrate(_observation(features=features))
    result = mapper.extract_matching_feature_voxels(torch.tensor([1.0, 0.0]), top_k=1)
    assert len(result.block_pool_idx) == 1
    assert len(result.block_scores) == 1
    assert float(result.block_scores[0]) > 0.9


def test_large_nominal_map_uses_lazy_sparse_capacity():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(512, 512, 512), max_blocks=100_000, block_size=8,
        voxel_size=0.02, origin=torch.tensor((-5.12, -5.12, -5.12)),
        feature_dim=32, feature_block_grid_size=2,
        feature_grid_height=1, feature_grid_width=1, device="cpu",
    ))
    data = integrator.tsdf.data
    assert data.block_data.shape[0] < data.max_blocks
    assert data.block_features.shape[0] < data.max_blocks
    assert int(data.num_allocated.item()) == 0
