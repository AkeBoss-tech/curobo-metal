"""CPU/MPS lifecycle checks for the portable kinematics loader."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.loader import KinematicsLoader, KinematicsLoaderCfg
from curobo._src.robot.types import CSpaceParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo.content import get_assets_path, get_robot_configs_path
from curobo.util_file import load_yaml


URDF = get_assets_path() / "robot/franka_description/franka_panda.urdf"


def _franka_config(*, device_cfg: DeviceCfg = DeviceCfg(), num_envs: int = 1):
    data = load_yaml(str(get_robot_configs_path() / "franka.yml"))["robot_cfg"]["kinematics"]
    return KinematicsLoaderCfg(**data, device_cfg=device_cfg, num_envs=num_envs)


def test_loader_compiles_packaged_relative_urdf_spheres_and_cspace():
    loader = KinematicsLoader(_franka_config(num_envs=3))

    assert loader.kinematics_parser.root_link == "base_link"
    assert loader.kinematics_config.num_dof == 7
    # The pinned Franka loader reserves eight disabled attachment spheres in
    # addition to its 61 physical ones.
    assert loader.kinematics_config.link_spheres.shape == (3, 69, 4)
    assert int((loader.kinematics_config.link_spheres[0, :, 3] < 0).sum()) == 8
    assert loader.self_collision_config.num_collision_checks > 0
    assert loader.get_joint_limits().joint_names == loader.joint_names
    assert loader.kinematics_config.cspace.joint_names == loader.joint_names
    torch.testing.assert_close(
        loader.kinematics_config.link_spheres[0], loader.kinematics_config.link_spheres[2]
    )


def test_loader_owns_config_inputs_and_reserves_unattached_collision_names():
    config = _franka_config()
    original_tools = config.tool_frames.copy()
    loader = KinematicsLoader(config)
    config.tool_frames.append("not_a_robot_link")
    config.self_collision_ignore["panda_hand"].append("not_a_robot_link")

    assert loader.kinematics_config.tool_frames == original_tools
    # Pinned Franka configuration reserves attached_object before an object is
    # installed.  The portable loader retains that future mutation state but
    # only compiles pairs for currently materialized robot links.
    assert "attached_object" in loader._robot.self_collision_buffer
    assert loader.self_collision_config.num_collision_checks > 0


def test_add_fixed_link_rebuilds_model_parser_and_tensor_cache():
    loader = KinematicsLoader(KinematicsLoaderCfg(
        base_link="base_link", tool_frames=["panda_hand"], urdf_path=str(URDF)
    ))
    before = loader.kinematics_config.num_links
    loader.add_fixed_link("portable_tool", "panda_hand")

    assert loader.kinematics_config.num_links == before + 1
    assert "portable_tool" in loader.kinematics_config.all_link_names
    assert loader.kinematics_parser.get_chain("base_link", "portable_tool")[-1] == "portable_tool"
    assert loader._build_chain("base_link", ["portable_tool"])[-1] == "portable_tool"
    with pytest.raises(ValueError, match="link already exists"):
        loader.add_fixed_link("portable_tool", "panda_hand")


def test_config_collision_disable_and_validation_boundaries():
    disabled = KinematicsLoaderCfg(
        base_link="base_link", tool_frames=["panda_hand"], urdf_path=str(URDF),
        collision_link_names=["panda_hand"], load_collision_spheres=False,
    )
    assert disabled.collision_spheres is None
    assert disabled.collision_link_names == []
    assert KinematicsLoader(disabled).total_spheres == 0

    with pytest.raises(ValueError, match="collision_link_names"):
        KinematicsLoaderCfg(
            base_link="base_link", tool_frames=["panda_hand"], urdf_path=str(URDF),
            collision_link_names=["panda_hand"], collision_spheres=None,
        )
    with pytest.raises(NotImplementedError, match="USD/Isaac"):
        KinematicsLoaderCfg(base_link="base", tool_frames=["tool"], urdf_path="robot.usd")


def test_cspace_config_is_materialized_without_aliasing_caller_tensors():
    names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5",
        "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2",
    ]
    cspace = CSpaceParams(
        joint_names=names,
        default_joint_position=[0.2, -0.3, *([0.0] * 7)],
        cspace_distance_weight=[1.0, 2.0, *([1.0] * 7)],
        null_space_weight=[3.0, 4.0, *([1.0] * 7)],
    )
    loader = KinematicsLoader(KinematicsLoaderCfg(
        base_link="panda_link0", tool_frames=["panda_link2"], urdf_path=str(URDF), cspace=cspace,
    ))
    assert loader.kinematics_config.cspace.default_joint_position[:2] == pytest.approx([0.2, -0.3])
    cspace.default_joint_position.add_(1.0)
    assert loader.kinematics_config.cspace.default_joint_position[:2] == pytest.approx([0.2, -0.3])


def test_prismatic_lock_reduces_active_cspace_and_preserves_lock_state(tmp_path):
    urdf = tmp_path / "slide.urdf"
    urdf.write_text(
        """<robot name="slide">
<link name="base"/><link name="slider"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/>
  <axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
<joint name="wrist" type="revolute"><parent link="slider"/><child link="tool"/>
  <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>""", encoding="utf-8"
    )
    loader = KinematicsLoader(KinematicsLoaderCfg(
        base_link="base", tool_frames=["tool"], urdf_path=str(urdf),
        lock_joints={"slide": 0.4},
        cspace={"joint_names": ["slide", "wrist"], "default_joint_position": [0.0, 0.1]},
    ))
    assert loader.joint_names == ["wrist"]
    assert loader.kinematics_config.num_dof == 1
    assert loader.lock_jointstate.joint_names == ["slide"]
    torch.testing.assert_close(loader.lock_jointstate.position, torch.tensor([0.4]))
    assert loader._robot.joints[0].xyz == pytest.approx((0.4, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match="nonzero revolute"):
        KinematicsLoader(KinematicsLoaderCfg(
            base_link="base", tool_frames=["tool"], urdf_path=str(urdf),
            lock_joints={"wrist": 0.1},
        ))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_loader_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"))
    loader = KinematicsLoader(_franka_config(device_cfg=device_cfg, num_envs=2))
    assert loader.kinematics_config.link_spheres.device.type == "mps"
    assert loader.self_collision_config.collision_pairs.device.type == "mps"
    assert loader.get_joint_limits().position.device.type == "mps"
