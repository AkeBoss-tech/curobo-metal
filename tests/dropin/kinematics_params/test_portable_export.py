"""Focused public behavior checks for portable ``KinematicsParams``."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest
import torch

from curobo._src.robot.kinematics import KinematicsCfg
from curobo._src.robot.types.kinematics_params import (
    CSpaceParams,
    DeviceCfg,
    JointLimits,
    JointState,
    JointType,
    KinematicsParams,
    RobotCollisionGeometry,
)


def _params() -> KinematicsParams:
    return KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config


def test_upstream_module_reexports_are_portable_and_importable():
    assert CSpaceParams is not None
    assert DeviceCfg is not None
    assert JointLimits is not None
    assert JointState is not None
    assert JointType is not None
    assert RobotCollisionGeometry is not None


def test_urdf_export_reflects_mutations_and_sphere_activation(tmp_path):
    params = _params()
    link_name = "panda_link1"
    params.update_link_mass(link_name, 2.5)
    params.update_link_com(link_name, torch.tensor([0.1, 0.2, 0.3]))
    params.update_link_inertia(link_name, torch.tensor([1, 2, 3, 0.1, 0.2, 0.3]))

    output = tmp_path / "portable.urdf"
    xml = params.export_to_urdf("portable_franka", str(output), include_spheres=True)
    assert output.read_text(encoding="utf-8") == xml
    root = ET.fromstring(xml)
    assert root.tag == "robot"
    assert root.attrib["name"] == "portable_franka"
    assert len(root.findall("link")) == params.num_links
    assert len(root.findall("joint")) == params.num_links - 1

    link = root.find(f"link[@name='{link_name}']")
    assert link is not None
    assert link.find("inertial/mass").attrib["value"] == "2.5"
    assert link.find("inertial/origin").attrib["xyz"] == "0.10000000149011612 0.20000000298023224 0.30000001192092896"
    assert link.find("inertial/inertia").attrib["izz"] == "3"
    active_spheres = int((params.link_spheres[0, :, 3] >= 0).sum().item())
    assert sum(len(item.findall("collision")) for item in root.findall("link")) == active_spheres

    params.disable_link_spheres(link_name)
    disabled = params.get_link_spheres(link_name)
    torch.testing.assert_close(disabled[:, 3], torch.full_like(disabled[:, 3], -100.0))
    root = ET.fromstring(params.export_to_urdf("portable_franka", include_spheres=True))
    exported = root.find(f"link[@name='{link_name}']")
    assert exported is not None
    assert not exported.findall("collision")


def test_cspace_defaults_and_unknown_sphere_link_validation():
    params = _params()
    cspace = params.robot_cfg.cspace
    cspace.default_joint_position = []
    cspace.cspace_distance_weight = None
    cspace.null_space_weight = None
    cspace.max_acceleration = None
    cspace.max_jerk = None
    params.load_cspace_cfg_from_kinematics()
    assert len(cspace.default_joint_position) == params.num_dof
    assert cspace.cspace_distance_weight == [1.0] * params.num_dof
    assert cspace.null_space_weight == [1.0] * params.num_dof
    assert cspace.max_acceleration == [10.0] * params.num_dof
    assert cspace.max_jerk == [500.0] * params.num_dof

    with pytest.raises(ValueError, match="unknown link"):
        params.get_sphere_index_from_link_name("not_a_robot_link")
