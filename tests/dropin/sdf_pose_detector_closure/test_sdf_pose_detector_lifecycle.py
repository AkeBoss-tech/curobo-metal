"""Lifecycle, tensor-boundary, and MPS closure tests for SDF pose refinement."""

import pytest
import torch

from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _mesh(device: str = "cpu") -> RobotMesh:
    vertices = torch.tensor(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), device=device)
    faces = torch.tensor(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), device=device)
    return RobotMesh(vertices, faces, device=device)


def _detector(*, device: str = "cpu", max_iterations: int = 5, inner_iterations: int = 2):
    return SDFPoseDetector(_mesh(device), SDFDetectorCfg(
        max_iterations=max_iterations,
        inner_iterations=inner_iterations,
        n_points=12,
        distance_threshold=2.0,
        device_cfg=DeviceCfg(device=device),
    ))


def _identity(device: str = "cpu", dtype: torch.dtype = torch.float32) -> Pose:
    return Pose(torch.zeros((1, 3), device=device, dtype=dtype),
                torch.tensor(((1.0, 0.0, 0.0, 0.0),), device=device, dtype=dtype))


def test_detector_canonicalizes_inputs_and_retains_an_isolated_state_snapshot():
    detector = _detector(max_iterations=0)
    points, _ = detector.robot_mesh.sample_surface_points(12)
    result = detector.detect_from_points(points.double(), initial_pose=_identity(dtype=torch.float64))
    assert result.pose.position.dtype == torch.float32
    assert result.n_iterations == 0
    assert detector.run_count == 1

    first = detector.last_refinement_state
    assert first is not None and first.position.dtype == torch.float32
    first.position.add_(1)
    second = detector.last_refinement_state
    assert second is not None
    assert not torch.equal(first.position, second.position)

    detector.reset()
    assert detector.last_refinement_state is None
    assert detector.run_count == 0


def test_detector_applies_inner_iteration_groups_without_discarding_the_remainder():
    detector = _detector(max_iterations=5, inner_iterations=2)
    points, _ = detector.robot_mesh.sample_surface_points(12)
    # Perfect alignment converges after the first complete inner group.  This
    # preserves the V2 outer/inner lifecycle rather than checking per step.
    result = detector.detect_from_points(points, initial_pose=_identity())
    assert result.n_iterations == 2
    assert detector.last_refinement_state.iterations == result.n_iterations


def test_detector_rejects_unsupported_configuration_and_invalid_boundaries():
    with pytest.raises(TypeError, match="float32"):
        SDFPoseDetector(_mesh(), SDFDetectorCfg(device_cfg=DeviceCfg(dtype=torch.float64)))
    with pytest.raises(ValueError, match="CPU and MPS"):
        SDFPoseDetector(_mesh(), SDFDetectorCfg(device_cfg=DeviceCfg(device=torch.device("xpu"))))

    detector = _detector(max_iterations=0)
    points, _ = detector.robot_mesh.sample_surface_points(4)
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        detector.detect_from_points(points.unsqueeze(0), initial_pose=_identity())
    with pytest.raises(ValueError, match="nonzero"):
        detector.detect_from_points(points, initial_pose=Pose(torch.zeros((1, 3)), torch.zeros((1, 4))))
    with pytest.raises(ValueError, match="finite"):
        detector.detect_from_points(points, config=torch.tensor([float("nan")]), initial_pose=_identity())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_detector_canonicalizes_cpu_pose_and_runs_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    detector = _detector(device="mps", max_iterations=2, inner_iterations=2)
    points, _ = detector.robot_mesh.sample_surface_points(12)
    result = detector.detect_from_points(points, initial_pose=_identity())
    state = detector.last_refinement_state
    assert result.pose.position.device.type == "mps"
    assert state is not None and state.best_JtJ.device.type == "mps"
