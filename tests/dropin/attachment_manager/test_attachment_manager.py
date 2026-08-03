"""Portable attachment-manager lifecycle and CPU/MPS contract coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _kinematics_and_manager(device: str = "cpu"):
    cfg = RobotSceneCollisionCfg.load_from_config(device_cfg=DeviceCfg(device=device))
    return cfg.kinematics, AttachmentManager(cfg.kinematics, device_cfg=cfg.device_cfg)


def _state(kinematics, batch: int = 1) -> JointState:
    q = kinematics.default_joint_position
    if batch > 1:
        q = q.expand(batch, -1).clone()
        q[1, 0] += 0.25
    return JointState.from_position(q, joint_names=kinematics.joint_names)


def _scene(device_cfg: DeviceCfg) -> SceneCollision:
    scene = SceneCfg(
        sphere=[
            Sphere("first", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.2),
            Sphere("second", pose=[1, 0, 0, 1, 0, 0, 0], radius=0.2),
        ]
    )
    return SceneCollision(SceneCollisionCfg(device_cfg, scene, cache={}))


def _active(scene: SceneCollision, name: str) -> bool:
    kind, slot = scene._names[0][name]
    assert kind == "analytic"
    return scene._analytic_active[0][slot]


def test_update_broadcasts_single_offset_to_multi_environment_bank():
    kinematics, manager = _kinematics_and_manager()
    params = kinematics.kinematics_config
    link = "panda_link7"
    original = params.get_link_spheres(link).clone()
    offset = Pose.from_list([0.5, 0.0, 0.4, 1, 0, 0, 0], DeviceCfg())

    manager.update(torch.tensor([[0.0, 0.0, 0.0, 0.02]]), _state(kinematics, 2), link, offset)

    written = params.link_spheres[:, params.get_sphere_index_from_link_name(link)[0]]
    assert params.num_envs == 2
    torch.testing.assert_close(written[:, 3], torch.full((2,), 0.02))
    assert not torch.allclose(written[0, :3], written[1, :3])
    manager.detach(link)
    torch.testing.assert_close(params.get_link_spheres(link, 0), original)


def test_attach_replacement_restores_prior_scene_obstacles():
    device_cfg = DeviceCfg()
    kinematics, _ = _kinematics_and_manager()
    scene = _scene(device_cfg)
    manager = AttachmentManager(kinematics, scene, device_cfg)
    state = _state(kinematics)
    payload = [Cuboid("payload", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.05] * 3)]

    manager.attach(state, payload, link_name="panda_link7", num_spheres=1, disable_obstacle_names=["first"])
    assert not _active(scene, "first")
    manager.attach(state, payload, link_name="panda_link7", num_spheres=1, disable_obstacle_names=["second"])

    assert _active(scene, "first")
    assert not _active(scene, "second")
    assert manager.attached_link_name == "panda_link7"
    manager.detach()
    assert _active(scene, "second")
    assert manager.attached_link_name is None


def test_attach_validates_all_scene_targets_before_mutating():
    device_cfg = DeviceCfg()
    kinematics, _ = _kinematics_and_manager()
    scene = _scene(device_cfg)
    manager = AttachmentManager(kinematics, scene, device_cfg)
    params = kinematics.kinematics_config
    original = params.get_link_spheres("panda_link7").clone()
    payload = [Cuboid("payload", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.05] * 3)]

    with pytest.raises(ValueError, match="does not exist"):
        manager.attach(
            _state(kinematics), payload, link_name="panda_link7", num_spheres=1,
            disable_obstacle_names=["first", "missing"],
        )

    assert _active(scene, "first")
    torch.testing.assert_close(params.get_link_spheres("panda_link7"), original)
    assert manager.attached_link_name is None


def test_invalid_offset_batch_does_not_resize_parameter_environments():
    kinematics, manager = _kinematics_and_manager()
    params = kinematics.kinematics_config
    offset = Pose(
        torch.zeros(3, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(3, -1).clone()
    )
    with pytest.raises(ValueError, match="one pose or one pose per environment"):
        manager.update(torch.tensor([[0.0, 0.0, 0.0, 0.02]]), _state(kinematics, 2), "panda_link7", offset)
    assert params.num_envs == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_attachment_update_runs_without_mps_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    kinematics, manager = _kinematics_and_manager("mps")
    offset = Pose.from_list([0.5, 0.0, 0.4, 1, 0, 0, 0], DeviceCfg(device="mps"))
    manager.update(
        torch.tensor([[0.0, 0.0, 0.0, 0.02]], device="mps"),
        _state(kinematics, 2),
        "panda_link7",
        offset,
    )
    assert manager.kinematics_params.link_spheres.device.type == "mps"
