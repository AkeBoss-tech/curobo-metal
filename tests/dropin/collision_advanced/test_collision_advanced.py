import pytest
import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.collision.wp_speed_metric import apply_speed_metric
from curobo._src.geom.types import Capsule, Cuboid, Cylinder, SceneCfg, Sphere
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _query(scene, values, env=None):
    tensor = torch.tensor(values, device=scene.device_cfg.device, dtype=scene.device_cfg.dtype)
    buffer = CollisionBuffer.from_shape(tensor.shape, scene.device_cfg)
    return scene.get_sphere_distance_raw(
        tensor, buffer, tensor.new_tensor(1.0), tensor.new_tensor(0.0), env
    ), buffer


def test_analytic_primitives_multi_env_and_mutation():
    cfg = DeviceCfg()
    scenes = [
        SceneCfg(sphere=[Sphere("ball", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5)]),
        SceneCfg(
            capsule=[
                Capsule(
                    "bar", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.2,
                    base=[0, 0, -1], tip=[0, 0, 1],
                )
            ],
            cylinder=[
                Cylinder("can", pose=[3, 0, 0, 1, 0, 0, 0], radius=0.5, height=2)
            ],
        ),
    ]
    scene = SceneCollision(SceneCollisionCfg(cfg, scenes, cache={}))
    values = [[[[1.0, 0, 0, 0.1]]], [[[0.5, 0, 0, 0.1]]]]
    distance, buffer = _query(scene, values, torch.tensor([0, 1]))
    torch.testing.assert_close(distance[:, 0, 0], torch.tensor([0.4, 0.2]))
    assert set(scene.collision_types) == {
        "cuboid", "mesh", "voxel", "sphere", "capsule", "cylinder"
    }
    scene.update_obstacle_pose(
        "ball", Pose.from_list([2, 0, 0, 1, 0, 0, 0], cfg), env_idx=0
    )
    moved, _ = _query(scene, values[:1])
    torch.testing.assert_close(moved[0, 0, 0], torch.tensor(0.4))
    scene.enable_obstacle("ball", False)
    disabled, _ = _query(scene, values[:1])
    assert torch.isinf(disabled).all()
    assert buffer.gradient.shape == torch.Size([2, 1, 1, 4])


def test_speed_metric_and_raw_warp_boundary():
    spheres = torch.zeros(1, 4, 1, 4)
    spheres[0, :, 0, 0] = torch.tensor([0.0, 1.0, 3.0, 6.0])
    distance = torch.ones(1, 4, 1)
    gradient = torch.ones(1, 4, 1, 4)
    out, out_gradient = apply_speed_metric(distance, gradient, spheres, torch.tensor(1.0))
    torch.testing.assert_close(out[0, 1:3, 0], torch.tensor([1.5, 2.5]))
    # The pinned kernel projects the gradient orthogonally to velocity before
    # scaling.  Motion is along x here, so x is removed while y/z scale with
    # speed and the unused fourth gradient component remains unchanged.
    torch.testing.assert_close(out_gradient[0, 1:3, 0, 0], torch.zeros(2))
    torch.testing.assert_close(out_gradient[0, 1:3, 0, 1:3], torch.tensor([[1.5, 1.5], [2.5, 2.5]]))
    torch.testing.assert_close(out_gradient[0, 1:3, 0, 3], torch.ones(2))
    from curobo._src.geom.collision.wp_autograd import SphereObstacleCollision
    with pytest.raises(NotImplementedError, match="Warp"):
        SphereObstacleCollision.apply()


def test_attachment_lifecycle_changes_production_spheres():
    config = RobotSceneCollisionCfg.load_from_config()
    kinematics = config.kinematics
    manager = AttachmentManager(kinematics)
    link = "panda_link7"
    original = kinematics.kinematics_config.get_link_spheres(link).clone()
    fitted = manager.fit_spheres(
        [Cuboid("payload", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])],
        num_spheres=1,
    )
    state = JointState.from_position(
        kinematics.default_joint_position, joint_names=kinematics.joint_names
    )
    manager.update(fitted, state, link_name=link)
    assert not torch.equal(
        original, kinematics.kinematics_config.get_link_spheres(link)
    )
    fk = kinematics.compute_kinematics(state)
    assert torch.isfinite(fk.robot_spheres).all()
    manager.detach(link)
    torch.testing.assert_close(
        original, kinematics.kinematics_config.get_link_spheres(link)
    )


def test_robot_scene_config_sampling_and_queries():
    scene = SceneCfg(
        sphere=[Sphere("far", pose=[10, 0, 0, 1, 0, 0, 0], radius=0.2)]
    )
    collision = RobotSceneCollision(
        RobotSceneCollisionCfg.load_from_config(scene_model=scene)
    )
    samples = collision.sample(3)
    assert samples.shape == (3, 7)
    assert collision.validate(samples).shape == (3,)
    state = collision.get_kinematics(samples[:, None, :])
    assert collision.get_collision_distance(state).shape[:2] == (3, 1)
    assert collision.get_self_collision(state).shape == (3, 1)
    distance, gradient = collision.get_collision_vector(state)
    assert distance.shape == (3, 1, state.robot_spheres.shape[-2])
    assert gradient.shape == state.robot_spheres.shape


def test_low_level_checker_routes_scene_data():
    cfg = DeviceCfg()
    data = SceneData.from_scene_model(
        SceneCfg(sphere=[
            Sphere("ball", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5)
        ]),
        cfg,
    )
    query = torch.tensor([[[[1.0, 0, 0, 0.1]]]])
    buffer = CollisionBuffer.from_shape(query.shape, cfg)
    result = CollisionChecker(cfg).get_sphere_distance(
        data, query, buffer, torch.tensor(1.0), torch.tensor(0.0)
    )
    torch.testing.assert_close(result, torch.tensor([[[0.4]]]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_advanced_scene_fallback_disabled_on_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = DeviceCfg(device="mps")
    scene = SceneCollision(SceneCollisionCfg(
        cfg,
        SceneCfg(cylinder=[
            Cylinder("can", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5, height=1)
        ]),
        cache={},
    ))
    distance, _ = _query(scene, [[[[1.0, 0, 0, 0.1]]]])
    assert distance.device.type == "mps"
    distance.sum().backward() if distance.requires_grad else None
