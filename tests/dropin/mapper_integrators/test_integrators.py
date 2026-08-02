"""Focused lifecycle coverage for the dense TSDF/ESDF integrator facades."""

import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
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


def _camera(device="cpu"):
    return CameraObservation(
        depth_image=torch.ones((4, 4), device=device),
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0)), device=device),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device),
        depth_to_meter=1.0,
    )


def test_tsdf_integrator_exports_source_shaped_mesh_tensors_and_stats():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(4, 4, 4), voxel_size=0.1, device="cpu",
    ))
    integrator.integrate(_camera())
    vertices, faces, normals, colors = integrator.extract_mesh_tensors()
    assert vertices.shape[-1] == normals.shape[-1] == colors.shape[-1] == 3
    assert faces.dtype == torch.int32
    assert colors.dtype == torch.uint8
    assert integrator.tsdf is integrator._tsdf
    stats = integrator.get_stats()
    assert stats["frame_count"] == 1
    assert stats["memory_mb"] > 0
    assert "last_camera_integration" in stats


def test_tsdf_time_decay_rebuilds_the_dense_derived_state():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, device="cpu",
        time_decay=0.5, minimum_tsdf_weight=0.8,
    ))
    state = integrator.mapper._mapper.state
    occupancy = torch.ones_like(state.occupancy)
    integrator.mapper._replace_state(
        weight=torch.ones_like(state.weight), occupancy=occupancy,
    )
    integrator._apply_frame_decay()
    decayed = integrator.mapper._mapper.state
    assert not decayed.occupancy.any()
    assert decayed.weight.eq(0).all()
    assert decayed.esdf.gt(0).all()


def test_tsdf_surface_export_has_source_tuple_shape_and_metric_distances():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, truncation_distance=0.2,
        minimum_tsdf_weight=0.5, device="cpu",
    ))
    state = integrator.mapper._mapper.state
    tsdf = torch.ones_like(state.tsdf)
    weight = torch.zeros_like(state.weight)
    tsdf[0, 0, 0, 0], weight[0, 0, 0, 0] = -0.25, 1.0
    tsdf[0, 1, 1, 1], weight[0, 1, 1, 1] = 0.75, 1.0
    integrator.mapper._replace_state(tsdf=tsdf, weight=weight)

    centers, colors, distances = integrator.extract_surface_voxels(sdf_threshold=0.1)
    assert centers.shape == (1, 3)
    assert colors.dtype == torch.uint8 and colors.tolist() == [[128, 128, 128]]
    assert distances.tolist() == pytest.approx([-0.05])
    # The bounded dense map retains V2 metric configuration introspection.
    assert integrator.voxel_size == 0.1
    assert integrator.truncation_distance == 0.2
    assert integrator.origin.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_tsdf_frustum_decay_only_applies_extra_decay_to_in_view_voxels():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(3, 3, 3), voxel_size=0.1, origin=torch.tensor((-0.15, -0.15, -0.15)),
        time_decay=1.0, frustum_decay=0.5, minimum_tsdf_weight=0.1, device="cpu",
    ))
    state = integrator.mapper._mapper.state
    integrator.mapper._replace_state(weight=torch.ones_like(state.weight))
    observation = _camera()
    # A wide field of view makes the small test volume unambiguously visible.
    observation.intrinsics = torch.tensor(((1.0, 0.0, 1.5), (0.0, 1.0, 1.5),
                                           (0.0, 0.0, 1.0)))
    integrator._apply_frame_decay(observation)
    weight = integrator.mapper._mapper.state.weight
    # Positive-z cells within the narrow camera image decay, while cells
    # behind the camera retain their temporal weight.
    assert (weight[0, :, :, 2] < 1).any()
    assert weight[0, :, :, 0].eq(1).all()


def test_tsdf_texture_export_uses_projective_rgbd_without_feature_volume():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, origin=torch.tensor((-0.1, -0.1, 0.9)),
        texture_num_cameras=1, texture_camera_image_height=4, texture_camera_image_width=4,
        truncation_distance=0.2, device="cpu",
    ))
    state = integrator.mapper._mapper.state
    integrator.mapper._replace_state(tsdf=torch.full_like(state.tsdf, -0.1),
                                    weight=torch.ones_like(state.weight))
    observation = _camera()
    observation.rgb_image = torch.tensor([17, 99, 201], dtype=torch.uint8).expand(4, 4, 3).clone()
    observation.depth_image = torch.ones((4, 4))
    observation.depth_to_meter = 1.0
    voxels = integrator.extract_occupied_voxels(texture_observations=observation)
    assert len(voxels) == 8
    assert voxels.texture_valid.any()
    assert voxels.colors_uint8()[voxels.texture_valid][0].tolist() == [17, 99, 201]


def test_tsdf_static_scene_stamping_executes_on_the_dense_backend():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(4, 4, 4), voxel_size=0.1, origin=torch.tensor((-0.2, -0.2, -0.2)),
        enable_static=True, device="cpu",
    ))
    scene = SceneCfg(cuboid=[Cuboid("box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.2, 0.2, 0.2])])
    assert integrator.update_static_obstacles(scene) > 0
    assert integrator.mapper._mapper.state.occupancy.any()


def test_esdf_integrator_keeps_real_site_and_voxel_grid_lifecycle():
    integrator = BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
        grid_shape=(4, 4, 4), esdf_grid_shape=(4, 4, 4), voxel_size=0.1,
        origin=torch.tensor((-0.2, -0.2, 0.8)), enable_static=True, device="cpu",
    ))
    integrator.integrate(_camera())
    integrator.update_static_obstacles(SceneCfg(cuboid=[Cuboid(
        "surface", pose=[0, 0, 1, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )]))
    field = integrator.compute_esdf()
    assert field.shape == (4, 4, 4)
    assert (integrator._site_index >= 0).any()
    grid = integrator.get_voxel_grid()
    assert grid.feature_tensor is field
    assert grid.dims == pytest.approx([0.4, 0.4, 0.4])
    assert integrator.get_stats()["total_memory_mb"] >= integrator.get_stats()["tsdf_memory_mb"]
    integrator.clear_region((-1, -1, -1), (1, 1, 1))
    assert integrator.dist_field.eq(0).all()
    assert integrator._site_index.eq(-1).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tsdf_projective_texture_and_surface_export_stay_on_mps():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, origin=torch.tensor((-0.1, -0.1, 0.9)),
        texture_num_cameras=1, texture_camera_image_height=4, texture_camera_image_width=4,
        truncation_distance=0.2, device="mps",
    ))
    state = integrator.mapper._mapper.state
    integrator.mapper._replace_state(tsdf=torch.full_like(state.tsdf, -0.1),
                                    weight=torch.ones_like(state.weight))
    observation = _camera("mps")
    observation.rgb_image = torch.full((4, 4, 3), 31, dtype=torch.uint8, device="mps")
    observation.depth_to_meter = 1.0
    voxels = integrator.extract_occupied_voxels(texture_observations=observation)
    centers, _colors, distances = integrator.extract_surface_voxels()
    assert voxels.texture_colors.device.type == centers.device.type == distances.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_integrator_mps_path_has_no_cpu_fallback():
    integrator = BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
        grid_shape=(4, 4, 4), esdf_grid_shape=(4, 4, 4), voxel_size=0.1, device="mps",
    ))
    integrator.integrate(_camera("mps"))
    assert integrator.compute_esdf().device.type == "mps"
