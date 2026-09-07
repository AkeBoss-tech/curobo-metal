"""Executable portable perception surface checks beyond raw-kernel boundaries."""

from curobo.types import DeviceCfg

import torch
import pytest

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
from curobo._src.perception.mapper.checkpoint_blocks import (
    BLOCK_CHECKPOINT_FORMAT,
    build_block_metadata,
    load_block_checkpoint,
    pack_hash_entry_host,
    rebuild_import_hash_state,
    save_block_checkpoint,
)
from curobo._src.perception.mapper.constants import PY_VALUE_MASK
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)


def _camera(depth):
    return CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0))),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu")),
        depth_to_meter=1.0,
    )


def test_filter_depth_uses_portable_bilateral_range_weighting_and_buffers():
    depth = torch.ones((1, 4, 4))
    depth[0, 1, 1] = 3.0
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
        Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu")), (4, 4),
    )
    assert depth.shape == valid.shape == (4, 4)
    assert normals.shape == (4, 4, 3)
    assert mapper.render_color_only(
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0))),
        Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu")), (4, 4),
    ).dtype == torch.uint8


def test_tsdf_integrator_and_mesh_detector_execute_without_warp():
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(grid_shape=(4, 4, 4), voxel_size=0.1, device="cpu"))
    integrator.integrate(_camera(torch.ones((4, 4))))
    assert integrator.get_stats()["observed_voxels"] > 0
    vertices = torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)))
    mesh = RobotMesh(vertices, torch.tensor(((0, 1, 2),)))
    points, normals = mesh.sample_surface_points(5)
    assert points.shape == normals.shape == (5, 3)
    result = PoseDetector(mesh, DetectorCfg(device_cfg=DeviceCfg("cpu"))).detect_from_points(points, None)
    assert result.alignment_error < 1e-5
    state = SDFRefinementState(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1))
    assert state.clone().copy_(state).iterations == 0


def test_high_level_esdf_facade_delegates_real_dense_mapping_operations():
    integrator = BlockSparseESDFIntegrator(BlockSparseESDFIntegratorCfg(
        grid_shape=(4, 4, 4), esdf_grid_shape=(4, 4, 4), voxel_size=0.1, device="cpu",
    ))
    integrator.integrate(_camera(torch.ones((4, 4))))
    field = integrator.compute_esdf()
    assert field.shape == (4, 4, 4)
    assert integrator.get_voxel_grid().feature_tensor.shape == (4, 4, 4)
    assert integrator.get_stats()["frame_count"] == 1
    assert integrator.clear_region((-1, -1, -1), (1, 1, 1)) > 0


def test_checkpoint_contract_roundtrip_and_warp_payload_boundary(tmp_path):
    mapper = Mapper(MapperCfg((0.2, 0.2, 0.2), voxel_size=0.1, device="cpu"))
    blocks = {key: value for key, value in mapper._mapper.state_dict().items() if isinstance(value, torch.Tensor)}
    checkpoint = tmp_path / "dense-blocks.pt"
    save_block_checkpoint(checkpoint, build_block_metadata(mapper.tsdf), blocks)
    loaded = load_block_checkpoint(checkpoint)
    assert loaded["format"] == BLOCK_CHECKPOINT_FORMAT
    assert torch.equal(loaded["blocks"]["tsdf"], blocks["tsdf"])
    entry = pack_hash_entry_host(-1, 0, 1, 7)
    assert isinstance(entry, int)
    assert (entry & PY_VALUE_MASK) == 7
    hash_table, block_to_hash_slot = rebuild_import_hash_state(
        torch.tensor([[0, 0, 0]], dtype=torch.int32), 4, 1
    )
    assert (hash_table != -1).sum() == 1
    assert block_to_hash_slot.tolist().count(-1) == 0
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(2, 2, 2), voxel_size=0.1, device="cpu",
    ))
    with pytest.raises(NotImplementedError, match="Warp block-pool"):
        integrator.import_blocks({"active_block_coords": torch.empty((0, 3), dtype=torch.int32)})
