import torch
import trimesh
import warp as wp

from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.pose import Pose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _franka() -> Kinematics:
    data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    links = data["robot_cfg"]["kinematics"]["mesh_link_names"]
    return Kinematics(KinematicsCfg.from_robot_yaml_file(data, tool_frames=links))


def test_rigid_robot_mesh_exposes_stable_warp_mesh_and_cached_samples() -> None:
    robot_mesh = RobotMesh.from_trimesh(trimesh.creation.box(), device="cpu")
    mesh = robot_mesh.mesh
    mesh_id = robot_mesh.mesh_id

    points, normals = robot_mesh.sample_surface_points(64)
    cached_points, cached_normals = robot_mesh.sample_surface_points(64)
    robot_mesh.update(torch.zeros(7))

    assert isinstance(mesh, wp.Mesh)
    assert robot_mesh.mesh is mesh
    assert robot_mesh.mesh_id == mesh_id
    assert torch.equal(points, cached_points)
    assert torch.equal(normals, cached_normals)
    assert torch.allclose(torch.linalg.vector_norm(normals, dim=1), torch.ones(64))


def test_articulated_robot_mesh_updates_vertices_without_replacing_bvh() -> None:
    robot_mesh = RobotMesh.from_kinematics(_franka(), device="cpu")
    initial = robot_mesh.vertices.clone()
    mesh = robot_mesh.mesh
    mesh_id = robot_mesh.mesh_id
    q = torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5])

    robot_mesh.update(q.unsqueeze(0))

    assert robot_mesh.n_vertices > 0
    assert robot_mesh.n_faces > 0
    assert not torch.allclose(robot_mesh.vertices, initial, atol=0.01)
    assert robot_mesh.current_joint_angles.shape == (7,)
    assert torch.equal(robot_mesh.current_joint_angles, q)
    assert robot_mesh.mesh is mesh
    assert robot_mesh.mesh_id == mesh_id


def test_articulated_surface_cache_tracks_updated_vertices() -> None:
    robot_mesh = RobotMesh.from_kinematics(_franka(), device="cpu")
    before, _ = robot_mesh.sample_surface_points(256)
    cached_faces = robot_mesh._sample_cache.face_indices.clone()

    robot_mesh.update(torch.tensor([0.4, -0.3, 0.2, -1.2, 0.3, 0.8, 0.4]))
    after, normals = robot_mesh.sample_surface_points(256)

    assert torch.equal(robot_mesh._sample_cache.face_indices, cached_faces)
    assert not torch.allclose(after, before, atol=0.01)
    assert torch.allclose(
        torch.linalg.vector_norm(normals, dim=1), torch.ones(256), atol=1e-5
    )
    assert len(robot_mesh.get_trimesh().faces) == robot_mesh.n_faces


def test_nested_surface_samples_preserve_identity_alignment_when_downsampled() -> None:
    robot_mesh = RobotMesh.from_trimesh(
        trimesh.creation.box(extents=[0.2, 0.2, 0.2]), device="cpu"
    )
    observed, _ = robot_mesh.sample_surface_points(5000)

    # The detector requests its configured model count after truncating the
    # larger observation. Nested sampling keeps those two surfaces paired.
    direct, _ = robot_mesh.sample_surface_points(500)
    assert torch.equal(observed[:500], direct)

    detector = SDFPoseDetector(robot_mesh, SDFDetectorCfg(n_points=500))
    identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    result = detector.detect_from_points(observed, initial_pose=identity)
    assert result.alignment_error < 0.01
