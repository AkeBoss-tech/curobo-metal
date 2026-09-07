from types import SimpleNamespace

import pytest
import torch

from curobo._src.geom.cv import get_projection_rays, project_depth_using_rays
from curobo._src.geom.mesh_triangulation import triangulate_mesh_faces
from curobo._src.geom.quaternion import angular_distance_axis_angle, quat_multiply
from curobo._src.geom.transform import (
    matrix_to_quaternion, pose_inverse, pose_multiply, quaternion_to_matrix,
    transform_points,
)
from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.scene import Capsule, Cuboid, Cylinder, Scene
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import CameraObservation, Pose
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.perception.pose_estimation.geometry import (
    ArticulatedRobotGeometry,
    RigidObjectGeometry,
)


def test_public_scene_and_sphere_fit():
    scene = Scene(
        cuboid=[Cuboid("box", [0,0,0,1,0,0,0], dims=[1,1,1])],
        capsule=[Capsule("cap", radius=.1, base=[0,0,0], tip=[0,0,1])],
        cylinder=[Cylinder("cyl", radius=.1, height=1.)],
    )
    assert len(scene) == 3

    class Mesh:
        vertices = torch.tensor([[0.,0,0],[1.,0,0],[0,1,0],[0,0,1]])

    result = fit_spheres_to_mesh(
        Mesh(), num_spheres=2, fit_type=SphereFitType.VOXEL,
        compute_metrics=True,
    )
    assert result.centers.shape == (2,3)
    assert result.metrics.num_spheres == 2


def test_transform_quaternion_and_camera_helpers_are_differentiable():
    q = torch.tensor([[1.,0,0,0]], requires_grad=True)
    p = torch.tensor([[1.,2,3.]], requires_grad=True)
    points = torch.tensor([[[1.,0,0.]]], requires_grad=True)
    transformed = transform_points(p, q, points)
    torch.testing.assert_close(transformed, torch.tensor([[[2.,2,3.]]]))
    transformed.sum().backward()
    assert q.grad is not None and points.grad is not None
    torch.testing.assert_close(matrix_to_quaternion(quaternion_to_matrix(q.detach())), q.detach())
    pi, qi = pose_inverse(p.detach(), q.detach())
    zero, identity = pose_multiply(p.detach(), q.detach(), pi, qi)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(identity, q.detach())
    assert angular_distance_axis_angle(q.detach(), q.detach()).item() == 0
    torch.testing.assert_close(quat_multiply(q.detach(), q.detach()), q.detach())

    intrinsics = torch.tensor([[10.,0,1.5],[0,10.,1.5],[0,0,1.]])
    rays = get_projection_rays(4, 4, intrinsics)
    assert project_depth_using_rays(torch.ones(4,4), rays).shape == (1,16,3)
    assert triangulate_mesh_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [0, 1, 2, 3],
        [4],
        DeviceCfg().cpu(),
    ) == [[1, 3, 0], [1, 2, 3]]


def test_depth_filter_and_mapper_lifecycle(tmp_path):
    depth = torch.ones(1, 4, 4)
    filtered, valid = FilterDepth((4,4), device="cpu")(depth)
    assert valid.all() and torch.equal(filtered, depth)

    cfg = MapperCfg((.2,.2,.2), voxel_size=.05, device="cpu", image_height=4, image_width=4)
    mapper = Mapper(cfg)
    camera = CameraObservation(
        depth_image=depth[0],
        intrinsics=torch.tensor([[10.,0,1.5],[0,10.,1.5],[0,0,1.]]),
        pose=Pose.from_list([0,0,-.2,1,0,0,0]),
        depth_to_meter=1.,
    )
    mapper.integrate(camera)
    assert mapper.get_stats()["observed_voxels"] > 0
    assert mapper.compute_esdf().feature_tensor.shape == cfg.grid_shape
    checkpoint = tmp_path / "blocks.pt"
    mapper.save_blocks(checkpoint)
    mapper.reset()
    assert mapper.import_blocks(checkpoint) > 0
    assert mapper.extract_mesh().vertices.shape[-1] == 3


def test_mps_fallback_disabled_when_available(monkeypatch):
    if not torch.backends.mps.is_available():
        return
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = MapperCfg((.2,.2,.2), voxel_size=.05, device="mps")
    mapper = Mapper(cfg)
    depth = torch.ones(4,4,device="mps")
    intrinsics = torch.tensor([[10.,0,1.5],[0,10.,1.5],[0,0,1.]],device="mps")
    pose = Pose.from_list([0,0,-.2,1,0,0,0]).to(device=torch.device("mps"))
    mapper.integrate(CameraObservation(depth_image=depth, intrinsics=intrinsics, pose=pose, depth_to_meter=1.))
    assert mapper._mapper.state.tsdf.device.type == "mps"


def test_rigid_geometry_samples_triangle_surfaces_instead_of_face_centroids():
    mesh = SimpleNamespace(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        faces=torch.tensor([[0, 1, 2]]),
    )
    torch.manual_seed(7)
    points, normals = RigidObjectGeometry(mesh).sample_surface_points(32)
    assert points.shape == normals.shape == (32, 3)
    torch.testing.assert_close(points[:, 0], torch.zeros(32))
    assert torch.all(points[:, 1:] >= 0)
    assert torch.all(points[:, 1:].sum(dim=-1) <= 1)
    torch.testing.assert_close(normals, torch.tensor([[1.0, 0.0, 0.0]]).expand_as(normals))


def test_articulated_geometry_caches_local_meshes_and_applies_fk_with_gradients():
    mesh = SimpleNamespace(
        vertices=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        faces=torch.tensor([[0, 1, 2]]),
        pose=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )

    class Robot:
        joint_names = ["slide"]
        mesh_link_names = ["tool"]

        @staticmethod
        def get_robot_link_meshes():
            return [mesh]

        @staticmethod
        def compute_kinematics(joint_state):
            x = joint_state.position[:, :1]
            position = torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
            return SimpleNamespace(
                tool_poses={"tool": Pose(position, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))}
            )

    geometry = ArticulatedRobotGeometry(Robot(), min_points_per_link=4, max_points_per_link=4)
    with pytest.raises(ValueError, match="Must call update"):
        geometry.sample_surface_points(2)
    q = torch.tensor([0.25], requires_grad=True)
    geometry.update(q)
    points, normals = geometry.sample_surface_points(4)
    assert points.shape == normals.shape == (4, 3)
    torch.testing.assert_close(points[:, 0], torch.full((4,), 1.25))
    torch.testing.assert_close(normals, torch.tensor([[1.0, 0.0, 0.0]]).expand_as(normals))
    points.sum().backward()
    assert q.grad is not None and q.grad.item() == 4.0
