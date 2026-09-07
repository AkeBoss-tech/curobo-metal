"""Regression coverage for portable perception tail parity."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _observation(depth: torch.Tensor, rgb: torch.Tensor) -> CameraObservation:
    count = 1 if depth.ndim == 2 else depth.shape[0]
    intrinsics = torch.tensor(
        ((4.0, 0.0, 3.5), (0.0, 4.0, 3.5), (0.0, 0.0, 1.0)),
        device=depth.device,
    ).expand(count, -1, -1).clone()
    pose = Pose(
        torch.zeros((count, 3), device=depth.device),
        torch.tensor(((1.0, 0.0, 0.0, 0.0),), device=depth.device).expand(count, -1).clone(),
    )
    if depth.ndim == 2:
        intrinsics = intrinsics[0]
    return CameraObservation(depth_image=depth, rgb_image=rgb, intrinsics=intrinsics, pose=pose)


def test_batched_sparse_rgb_is_a_weighted_camera_accumulator() -> None:
    config = BlockSparseTSDFIntegratorCfg(
        grid_shape=(512, 512, 512), max_blocks=32, voxel_size=0.02,
        origin=torch.zeros(3), num_cameras=2, image_height=8, image_width=8,
        device="cpu",
    )
    integrator = BlockSparseTSDFIntegrator(config)
    depth = torch.ones((2, 8, 8))
    rgb = torch.zeros((2, 8, 8, 3), dtype=torch.uint8)
    rgb[0, ..., 0] = 255
    rgb[1, ..., 2] = 255
    integrator.integrate(_observation(depth, rgb))
    data = integrator.tsdf.data
    active = data.block_grid_rgb[:, 0, 3] > 0
    normalized = data.block_grid_rgb[active, 0, :3].float() / data.block_grid_rgb[
        active, 0, 3:4
    ].float()
    torch.testing.assert_close(normalized, normalized.new_tensor((0.5, 0.0, 0.5)).expand_as(normalized))


def test_texture_projector_is_initialized_and_renders_missing_depth() -> None:
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(512, 512, 512), max_blocks=8, voxel_size=0.02,
        origin=torch.zeros(3), texture_num_cameras=2,
        image_height=8, image_width=8, device="cpu",
    ))
    rgb = torch.zeros((8, 8, 3), dtype=torch.uint8)
    first = _observation(torch.ones((8, 8)), rgb)
    second = _observation(torch.ones((8, 8)), rgb)
    second.depth_image = None
    batches = integrator._texture_projector._normalize_projective_texture_observations(
        (first, second)
    )
    assert len(batches) == 1
    torch.testing.assert_close(batches[0][1][0], torch.ones((8, 8)))
    assert torch.count_nonzero(batches[0][1][1]) == 0


def test_renderer_preserves_camera_batch_shape() -> None:
    integrator = BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(16, 16, 16), max_blocks=8, voxel_size=0.05,
        origin=torch.tensor((-0.4, -0.4, 0.0)), image_height=8, image_width=8,
        device="cpu",
    ))
    observation = _observation(torch.ones((8, 8)), torch.zeros((8, 8, 3), dtype=torch.uint8))
    integrator.integrate(observation)
    intrinsics = observation.intrinsics.expand(2, -1, -1).clone()
    pose = Pose(
        observation.pose.position.expand(2, -1).clone(),
        observation.pose.quaternion.expand(2, -1).clone(),
    )
    depth, normals, valid = BlockSparseTSDFRenderer(integrator).render(intrinsics, pose, (8, 8))
    assert depth.shape == (2, 8, 8)
    assert normals.shape == (2, 8, 8, 3)
    assert valid.shape == (2, 8, 8)
    torch.testing.assert_close(depth[0], depth[1])


def test_sdf_inner_executor_replays_current_state() -> None:
    vertices = torch.tensor(((-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.0, 0.1, 0.0)))
    mesh = RobotMesh(vertices, torch.tensor(((0, 1, 2),)), device="cpu")
    points, _ = mesh.sample_surface_points(32)
    points = points + torch.tensor((0.02, -0.01, 0.0))
    detector = SDFPoseDetector(mesh, SDFDetectorCfg(
        inner_iterations=2, max_iterations=4, n_points=32, distance_threshold=1.0,
        use_cuda_graph=False,
    ))
    identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    state = detector._setup_refinement(points, identity)
    initial = state.best_error.clone()
    state = detector._refine_inner_executor(state)
    assert state.best_error < initial


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_sdf_default_config_inherits_mps_mesh_device() -> None:
    vertices = torch.tensor(((-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.0, 0.1, 0.0)))
    mesh = RobotMesh(vertices, torch.tensor(((0, 1, 2),)), device="mps")
    detector = SDFPoseDetector(mesh)
    points, _ = mesh.sample_surface_points(16)
    result = detector.detect_from_points(
        points, initial_pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    )
    assert result.pose.position.device.type == "mps"
