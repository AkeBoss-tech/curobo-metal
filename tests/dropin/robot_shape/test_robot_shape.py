"""Executable portable contracts for the pinned robot-model surface."""

from __future__ import annotations

import pytest
import torch

from curobo.content import get_assets_path
from curobo._src.robot.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.loader import KinematicsLoader, KinematicsLoaderCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


URDF = get_assets_path() / "robot/franka_description/franka_panda.urdf"


def test_kinematics_params_tensor_views_mutation_and_independent_clone():
    config = KinematicsCfg.from_robot_yaml_file("franka.yml")
    params = config.kinematics_config
    assert params.joint_limits.position.shape == (2, params.num_dof)
    assert params.joint_map_type.shape[0] == len(params.robot_cfg.joints)
    assert params.tool_frame_map.tolist() == [params.link_name_to_idx_map["panda_hand"]]
    assert params.fixed_transforms.shape == (params.num_links, 4, 4)
    assert params.n_tree_levels >= 2 and params.max_level_width >= 1

    link = "panda_link1"
    before = params.get_link_spheres(link).clone()
    params.disable_link_spheres(link)
    assert bool((params.get_link_spheres(link)[..., 3] <= 0).all())
    params.enable_link_spheres(link)
    torch.testing.assert_close(params.get_link_spheres(link), before)
    cloned = params.clone()
    cloned.update_link_mass(link, 123.0)
    assert params.get_link_masses_com(link)[-1] != cloned.get_link_masses_com(link)[-1]


def test_arbitrary_link_pose_and_params_recompile_are_differentiable():
    config = KinematicsCfg.from_robot_yaml_file("franka.yml")
    model = Kinematics(config, compute_jacobian=True)
    q = model.default_joint_position.repeat(2, 1).requires_grad_()
    pose = model.get_link_poses(q, ["panda_link2", "panda_hand"])
    assert pose.position.shape == (2, 2, 3)
    assert pose.quaternion.shape == (2, 2, 4)
    assert torch.autograd.grad(pose.position.square().sum(), q)[0].shape == q.shape
    model.update_kinematics_config(config.kinematics_config.clone())
    state = model.compute_kinematics(JointState.from_position(q, model.joint_names))
    assert state.tool_poses.position.shape == (2, 1, 1, 3)


def test_loader_self_collision_pairs_and_cfg_factory():
    loader_cfg = KinematicsLoaderCfg(
        base_link="base_link", urdf_path=str(URDF), tool_frames=["panda_hand"],
        device_cfg=DeviceCfg(),
    )
    loader = KinematicsLoader(loader_cfg)
    self_collision = loader.self_collision_config
    assert self_collision.num_spheres == 0
    config = KinematicsCfg.from_config(loader_cfg)
    assert config.dof == loader.kinematics_config.num_dof


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_robot_param_views_and_arbitrary_link_fk_stay_on_mps(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    config = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"))
    )
    model = Kinematics(config)
    q = model.default_joint_position.unsqueeze(0)
    pose = model.get_link_poses(q, ["panda_link2"])
    assert pose.position.device.type == "mps"
    assert config.kinematics_config.joint_limits.position.device.type == "mps"
