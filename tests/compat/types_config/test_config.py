from __future__ import annotations

from pathlib import Path

import pytest
import torch

from curobo_metal.config import (
    RobotCfg,
    UnsupportedConfigError,
    load_robot_config,
    load_world_config,
)
from curobo_metal.config.loaders import _parse_yaml


FIXTURES = Path(__file__).parent / "fixtures"


def test_dependency_free_yaml_subset_parser() -> None:
    value = _parse_yaml((FIXTURES / "tiny_robot.yml").read_text())
    spheres = value["robot_cfg"]["kinematics"]["collision_spheres"]
    assert spheres["arm"][0] == {"center": [0.1, 0.0, 0.0], "radius": 0.08}


def test_dependency_free_yaml_parser_accepts_indentless_sequences() -> None:
    value = _parse_yaml(
        """
robot_cfg:
  kinematics:
    tool_frames:
    - panda_hand
    - panda_finger
    base_link: panda_link0
"""
    )
    assert value["robot_cfg"]["kinematics"] == {
        "tool_frames": ["panda_hand", "panda_finger"],
        "base_link": "panda_link0",
    }


def test_curobo_content_asset_root_resolves_from_configs_tree(tmp_path: Path) -> None:
    content = tmp_path / "content"
    config = content / "configs/robot/example.yml"
    urdf = content / "assets/robot/example/robot.urdf"
    config.parent.mkdir(parents=True)
    urdf.parent.mkdir(parents=True)
    urdf.write_text(
        "<robot name='example'><link name='base'/><link name='tool'/>"
        "<joint name='joint' type='fixed'><parent link='base'/>"
        "<child link='tool'/></joint></robot>"
    )
    config.write_text(
        "robot_cfg:\n"
        "  kinematics:\n"
        "    urdf_path: robot/example/robot.urdf\n"
        "    asset_root_path: robot/example\n"
        "    base_link: base\n"
        "    tool_frames:\n"
        "    - tool\n"
    )
    robot = RobotCfg.create(config)
    assert robot.name == "example"
    assert robot.tool_frames == ["tool"]


def test_explicit_urdf_base_selects_descendant_subtree(tmp_path: Path) -> None:
    urdf = tmp_path / "subtree.urdf"
    urdf.write_text(
        "<robot name='subtree'>"
        "<link name='world'/><link name='base'/><link name='tool'/>"
        "<joint name='mount' type='fixed'><parent link='world'/>"
        "<child link='base'/></joint>"
        "<joint name='arm' type='revolute'><parent link='base'/>"
        "<child link='tool'/><axis xyz='0 0 1'/>"
        "<limit lower='-1' upper='1' velocity='2' effort='3'/></joint>"
        "</robot>"
    )
    robot = load_robot_config(
        {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": str(urdf),
                    "base_link": "base",
                    "tool_frames": ["tool"],
                }
            }
        }
    )
    assert [link.name for link in robot.links] == ["base", "tool"]
    assert [joint.name for joint in robot.joints] == ["arm"]


def test_yaml_urdf_preserves_tree_metadata_and_compiles() -> None:
    robot = load_robot_config(FIXTURES / "tiny_robot.yml")
    assert robot.name == "tiny_tree"
    assert [link.name for link in robot.links] == ["base", "arm", "tip", "finger"]
    assert [joint.name for joint in robot.joints] == [
        "shoulder", "tip_fixed", "finger_mimic"
    ]
    shoulder, _, mimic = robot.joints
    assert shoulder.axis == (0.0, 0.0, 2.0)
    assert vars(shoulder.limits) == {
        "lower": -1.5, "upper": 1.25, "velocity": 2.5, "effort": 12.0
    }
    assert mimic.mimic_joint == "shoulder"
    assert (mimic.mimic_multiplier, mimic.mimic_offset) == (-0.02, 0.04)
    arm = robot.links[1]
    assert arm.mass == 2.0
    assert arm.com == (0.1, 0.0, 0.0)
    assert arm.inertia == (0.2, 0.3, 0.4, 0.01, 0.02, 0.03)
    tree = robot.to_tree_robot()
    assert tree.joint_names == ("shoulder",)
    assert tree.links[3].q_index == 0
    assert tree.links[3].multiplier == -0.02
    model = robot.to_whole_body_model(device="cpu", dtype=torch.float64)
    assert model.dof == 1 and model.link_names == ("base", "arm", "tip", "finger")
    spheres, indices = robot.to_collision_inputs(device="cpu", dtype=torch.float64)
    assert spheres.shape == (2, 4)
    assert indices.tolist() == [0, 1]


def test_robotcfg_mapping_round_trip(tmp_path: Path) -> None:
    original = RobotCfg.create(FIXTURES / "tiny_robot.yml")
    target = tmp_path / "round_trip.yml"
    original.write_config(target)
    # Serialized URDF path is absolute so the portable mapping remains relocatable.
    recovered = RobotCfg.create(target)
    assert recovered.joint_names == original.joint_names
    assert recovered.cspace.default_joint_position == [0.25]
    torch.testing.assert_close(
        recovered.to_collision_inputs()[0],
        original.to_collision_inputs()[0],
    )


def test_xrdf_overrides_and_preserves_disabled_sphere() -> None:
    robot = load_robot_config(FIXTURES / "tiny_xrdf_robot.yml")
    assert robot.tool_frames == ["tip"]
    assert robot.cspace.default_joint_position == [-0.2]
    assert robot.cspace.max_acceleration == [4.0]
    assert [sphere.radius for sphere in robot.collision_spheres] == [0.09, -0.01]
    assert robot.self_collision_buffer == {"arm": 0.02}


def test_plain_urdf_tree_and_serial_boundary() -> None:
    robot = RobotCfg.from_basic(
        FIXTURES / "tiny_tree.urdf", "base", ["tip", "finger"],
        device_cfg=robot_device(),
    )
    assert robot.device_cfg.dtype == torch.float64
    with pytest.raises(UnsupportedConfigError, match="exactly one tool frame"):
        robot.to_serial_robot()


def test_serial_conversion_selects_single_tool_path_from_tree() -> None:
    robot = RobotCfg.create(FIXTURES / "tiny_robot.yml")
    robot.tool_frames = ["tip"]
    serial = robot.to_serial_robot()
    assert serial.dof == 1
    assert [joint.name for joint in serial.joints] == ["arm", "tip"]


def robot_device():
    from curobo_metal.types import DeviceCfg
    return DeviceCfg("cpu", torch.float64)


def test_world_and_exact_unsupported_boundaries(tmp_path: Path) -> None:
    world = load_world_config({
        "world_cfg": {
            "cuboid": [{"name": "table", "pose": [0, 0, 0, 1, 0, 0, 0], "dims": [1, 2, 0.1]}],
            "sphere": [{"name": "ball", "pose": [0, 0, 1, 1, 0, 0, 0], "radius": 0.2}],
        }
    })
    assert world.cuboid[0].name == "table" and world.sphere[0].radius == 0.2
    with pytest.raises(UnsupportedConfigError, match="cuboid/sphere only"):
        load_world_config({"mesh": [{"file_path": "mesh.obj"}]})
    usd = tmp_path / "robot.usd"
    usd.write_text("#usda 1.0")
    with pytest.raises(UnsupportedConfigError, match="USD/Isaac"):
        load_robot_config(usd)
    with pytest.raises(UnsupportedConfigError, match="unsupported URDF type"):
        bad = tmp_path / "bad.urdf"
        bad.write_text(
            "<robot name='bad'><link name='a'/><link name='b'/>"
            "<joint name='j' type='planar'><parent link='a'/><child link='b'/></joint></robot>"
        )
        load_robot_config(bad)
