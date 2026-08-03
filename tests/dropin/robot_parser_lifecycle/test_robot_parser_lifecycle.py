"""Portable lifecycle coverage for the pinned URDF parser surface."""

from __future__ import annotations

import math

import numpy as np
import pytest

from curobo._src.geom.types import Cuboid, Cylinder, Sphere
from curobo._src.robot.parser import UrdfRobotParser
from curobo._src.robot.types import JointType, LinkParams


def _write(tmp_path, text: str):
    path = tmp_path / "robot.urdf"
    path.write_text(text, encoding="utf-8")
    return path


def test_parser_preserves_parent_records_and_extra_link_overlay(tmp_path):
    path = _write(tmp_path, """<robot name="tree">
      <link name="base"/><link name="arm"/><link name="tool"/>
      <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="2" effort="3"/></joint>
      <joint name="tool_fixed" type="fixed"><parent link="arm"/><child link="tool"/></joint>
    </robot>""")
    extra = LinkParams("camera", "camera_joint", JointType.FIXED, np.eye(4)[:3], parent_link_name="tool")
    parser = UrdfRobotParser(str(path), extra_links={"camera": extra})

    assert parser._parent_map["arm"] == {"parent": "base", "jid": 0, "joint_name": "shoulder"}
    assert parser._parent_map["camera"]["parent"] == "tool"
    assert parser.get_chain("base", "camera") == ["base", "arm", "tool", "camera"]
    assert parser.get_actuated_joint_names() == ["shoulder"]
    assert parser._get_from_extra_links("missing") is None
    assert parser.get_link_parameters("camera") is extra


def test_continuous_negative_axis_and_inertial_defaults_match_portable_contract(tmp_path):
    path = _write(tmp_path, """<robot name="continuous">
      <link name="base"/><link name="arm"><inertial><mass value="0"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
        <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
      <joint name="spin" type="continuous"><parent link="base"/><child link="arm"/>
        <axis xyz="0 0 -1"/></joint>
    </robot>""")
    params = UrdfRobotParser(str(path)).get_link_parameters("arm")

    assert params.joint_type is JointType.Z_ROT
    assert params.joint_axis.tolist() == [0.0, 0.0, 1.0]
    assert params.joint_offset == [-1.0, 0.0]
    assert params.joint_limits == pytest.approx([-2 * math.pi, 2 * math.pi])
    assert params.joint_velocity_limits == [-100.0, 100.0]
    assert params.joint_effort_limit == [100.0]
    assert params.link_mass == pytest.approx(0.01)
    np.testing.assert_allclose(params.link_com, [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(params.link_inertia, [1e-6, 1e-6, 1e-6, 0, 0, 0])


def test_primitive_geometry_preserves_full_origin_pose_and_collision_selection(tmp_path):
    path = _write(tmp_path, """<robot name="geometry">
      <link name="base"><visual name="box"><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
        <geometry><box size="2 4 6"/></geometry></visual>
        <collision name="cyl"><origin xyz="3 2 1"/><geometry><cylinder radius="0.2" length="0.6"/></geometry></collision>
        <collision name="sphere"><origin xyz="0 1 0" rpy="0.1 0.2 0.3"/><geometry><sphere radius="0.4"/></geometry></collision>
      </link></robot>""")
    parser = UrdfRobotParser(str(path))
    visual = parser.get_link_geometry("base")
    collision = parser.get_link_geometry("base", use_collision_mesh=True)

    assert len(visual) == 1 and isinstance(visual[0], Cuboid)
    assert visual[0].dims == [2.0, 4.0, 6.0]
    assert visual[0].pose[:3] == [1.0, 2.0, 3.0]
    assert visual[0].pose[3:] != [1.0, 0.0, 0.0, 0.0]
    assert [type(item) for item in collision] == [Cylinder, Sphere]
    assert collision[0].pose[:3] == [3.0, 2.0, 1.0]
    assert collision[1].pose[:3] == [0.0, 1.0, 0.0]


def test_mesh_remains_explicit_optional_backend_boundary(tmp_path):
    path = _write(tmp_path, """<robot name="mesh"><link name="base"><visual>
      <geometry><mesh filename="package://description/mesh.stl" scale="1 2 3"/></geometry>
    </visual></link></robot>""")
    parser = UrdfRobotParser(str(path), mesh_root=str(tmp_path / "assets"))
    assert str(tmp_path / "assets" / "description" / "mesh.stl") == parser._file_name_handler("package://description/mesh.stl")
    with pytest.raises(NotImplementedError, match="file-backed URDF mesh"):
        parser.get_link_geometry("base")
    with pytest.raises(NotImplementedError, match="file-backed URDF mesh"):
        parser.get_link_mesh("base")


def test_links_without_meshes_return_none_from_mesh_accessor(tmp_path):
    path = _write(tmp_path, """<robot name="primitive"><link name="base"><visual>
      <geometry><sphere radius="0.1"/></geometry></visual></link></robot>""")
    assert UrdfRobotParser(str(path)).get_link_mesh("base") is None
