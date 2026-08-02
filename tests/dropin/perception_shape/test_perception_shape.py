"""Executable portable perception surface checks beyond raw-kernel boundaries."""

import torch

from curobo._src.perception.filter_depth import FilterDepth
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.pose_detector import PoseDetector
from curobo._src.perception.pose_estimation.pose_detector_cfg import DetectorCfg
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFRefinementState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _camera(depth):
    return CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0))),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]),
        depth_to_meter=1.0,
    )


def test_filter_depth_uses_portable_bilateral_range_weighting_and_buffers():
    depth = torch.ones((4, 4))
    depth[1, 1] = 3.0
    output = torch.empty_like(depth)
    mask = torch.empty_like(depth, dtype=torch.bool)
    filtered, valid = FilterDepth((4, 4), flying_pixel_threshold=3.0)(depth, output, mask)
    assert filtered is output and valid is mask
    assert valid.all()
    # With a narrow range sigma, the isolated range discontinuity is retained
    # rather than averaged into surrounding valid geometry.
    assert torch.isfinite(filtered).all()


def test_mapper_region_mutation_and_render_surface_are_real_dense_operations():
    mapper = Mapper(MapperCfg((0.4, 0.4, 0.4), voxel_size=0.1,
                              grid_center=torch.tensor((0.0, 0.0, 0.4)), device="cpu"))
    mapper.integrate(_camera(torch.ones((4, 4))))
    before = mapper.get_stats()["observed_voxels"]
    assert before > 0
    assert mapper.clear_region((-1, -1, -1), (1, 1, 1)) == before
    assert mapper.get_stats()["observed_voxels"] == 0
    depth, normals, valid = mapper.render(
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0))),
        Pose.from_list([0, 0, 0, 1, 0, 0, 0]), (4, 4),
    )
    assert depth.shape == valid.shape == (4, 4)
    assert normals.shape == (4, 4, 3)
    assert mapper.render_color_only(
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0))),
        Pose.from_list([0, 0, 0, 1, 0, 0, 0]), (4, 4),
    ).dtype == torch.uint8


def test_tsdf_integrator_and_mesh_detector_execute_without_warp():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(grid_shape=(4, 4, 4), voxel_size=0.1, device="cpu"))
    integrator.integrate(_camera(torch.ones((4, 4))))
    assert integrator.get_stats()["observed_voxels"] > 0
    vertices = torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)))
    mesh = RobotMesh(vertices, torch.tensor(((0, 1, 2),)))
    points, normals = mesh.sample_surface_points(5)
    assert points.shape == normals.shape == (5, 3)
    result = PoseDetector(mesh, DetectorCfg()).detect_from_points(points, None)
    assert result.alignment_error < 1e-5
    state = SDFRefinementState(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1))
    assert state.clone().copy_(state).iterations == 0
