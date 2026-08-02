"""High-level portable pose-refinement behavior, without Warp kernels."""

import pytest
import torch

from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import (
    SDFPoseDetector,
    SDFRefinementState,
)
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _camera(depth: torch.Tensor, pose: Pose) -> CameraObservation:
    return CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5), (0.0, 0.0, 1.0)),
                                device=depth.device, dtype=depth.dtype),
        pose=pose,
        depth_to_meter=1.0,
    )


def _mesh(device="cpu") -> RobotMesh:
    vertices = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device)
    faces = torch.tensor(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), device=device)
    return RobotMesh(vertices, faces, device=device)


def test_sdf_detector_recovers_a_local_rigid_transform_with_its_sdf_config():
    mesh = _mesh()
    model, _ = mesh.sample_surface_points(64)
    # This is deliberately a local registration API: the CUDA global rotation
    # sampler is a raw Warp boundary, while a 25 degree/3 cm perturbation is a
    # realistic refinement basin for the portable tensor solver.
    truth = Pose.from_euler_xyz(torch.tensor((0.0, 0.0, 0.25)), torch.tensor((0.02, -0.03, 0.03)))
    observed = truth.transform_points(model)
    result = SDFPoseDetector(mesh, SDFDetectorCfg(
        max_iterations=24, n_points=64, distance_threshold=2.0,
        convergence_threshold=1e-6, rotation_convergence_threshold=1e-6,
    )).detect_from_points(observed, initial_pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
    assert result.config is None
    assert result.confidence == 1.0
    assert result.alignment_error < 2e-4
    torch.testing.assert_close(result.pose.position, truth.position, atol=2e-5, rtol=0)
    assert result.pose.angular_distance(truth).item() < 2e-4


def test_sdf_detector_requires_local_pose_and_state_is_cloneable():
    mesh = _mesh()
    points, _ = mesh.sample_surface_points(4)
    detector = SDFPoseDetector(mesh, SDFDetectorCfg(max_iterations=0, n_points=4))
    with pytest.raises(ValueError, match="initial_pose"):
        detector.detect_from_points(points)
    state = SDFRefinementState(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(()),
                               observed_points=points, best_n_valid=torch.tensor(4))
    copied = state.clone()
    assert copied.copy_(state).best_n_valid.item() == 4
    assert copied.observed_points.data_ptr() != state.observed_points.data_ptr()


def test_sdf_detector_exposes_pinned_refinement_state_lifecycle_without_warp():
    mesh = _mesh()
    points, _ = mesh.sample_surface_points(12)
    detector = SDFPoseDetector(mesh, SDFDetectorCfg(
        max_iterations=2, inner_iterations=2, n_points=12, distance_threshold=1.0,
    ))
    initial = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    state = detector._setup_refinement(points, initial)
    assert state.n_points == 12
    assert state.best_JtJ.shape == (6, 6)
    assert state.best_Jtr.shape == (6,)
    assert state.best_sum_sq.ndim == 0
    assert state.best_n_valid.item() == 12
    # The upstream named buffer construction is accepted independently of the
    # compact portable constructor and retains deep-copy semantics.
    named = SDFRefinementState(
        observed_points=state.observed_points, n_points=state.n_points,
        best_position=state.best_position, best_quaternion=state.best_quaternion,
        best_error=state.best_error, best_sum_sq=state.best_sum_sq,
        best_n_valid=state.best_n_valid, best_JtJ=state.best_JtJ, best_Jtr=state.best_Jtr,
        lambda_damping=state.lambda_damping, translation_change=state.translation_change,
        rotation_change=state.rotation_change,
    )
    copied = named.clone()
    assert copied.observed_points.data_ptr() != named.observed_points.data_ptr()
    updated = detector._refine_inner_iterations(copied)
    assert updated.iterations == detector.config.inner_iterations
    assert torch.isfinite(updated.best_error)


def test_dense_tsdf_refiner_returns_curobo_pose_error_iterations_contract():
    mapper = Mapper(MapperCfg((0.4, 0.4, 0.4), voxel_size=0.1,
                              grid_center=torch.tensor((0.0, 0.0, 0.4)),
                              depth_minimum_distance=0.1, depth_maximum_distance=2.0,
                              device="cpu"))
    identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    depth = torch.ones((4, 4))
    mapper.integrate(_camera(depth, identity))
    refiner = BlockSparseRaycastPoseRefiner(mapper, BlockSparseRaycastRefinerCfg(
        max_iterations=2, minimum_valid_depth_pixels=4, depth_maximum_distance=2.0,
    ))
    pose, error, iterations = refiner.refine_pose(depth, _camera(depth, identity).intrinsics, identity)
    assert isinstance(pose, Pose)
    assert isinstance(error, float) and error >= 0
    assert iterations == 2
    torch.testing.assert_close(pose.position, identity.position)


def test_refiner_rejects_depth_without_its_required_valid_observations():
    mapper = Mapper(MapperCfg((0.2, 0.2, 0.2), voxel_size=0.1, device="cpu"))
    refiner = BlockSparseRaycastPoseRefiner(mapper, BlockSparseRaycastRefinerCfg(minimum_valid_depth_pixels=4))
    with pytest.raises(ValueError, match="fewer valid"):
        refiner.refine_pose(torch.zeros((2, 2)), torch.eye(3), Pose.from_list([0, 0, 0, 1, 0, 0, 0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_sdf_detector_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    mesh = _mesh("mps")
    points, _ = mesh.sample_surface_points(12)
    result = SDFPoseDetector(mesh, SDFDetectorCfg(max_iterations=2, n_points=12,
                                                   distance_threshold=1.0,
                                                   device_cfg=DeviceCfg(device="mps"))).detect_from_points(
        points, initial_pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0], DeviceCfg(device="mps"))
    )
    assert result.pose.position.device.type == "mps"
