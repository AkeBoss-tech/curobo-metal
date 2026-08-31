from pathlib import Path
from xml.etree import ElementTree

import pytest

from curobo.content import (
    get_assets_path,
    get_robot_path,
    get_scene_configs_path,
    list_available_robots,
)
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)


def test_pinned_franka_config_resolves_complete_real_asset_tree() -> None:
    assert list_available_robots() == [
        "dual_ur10e", "franka", "simple_mimic_robot", "unitree_g1", "ur10e"
    ]
    config_path = get_robot_path("franka")
    assert config_path == get_robot_configs_path() / "franka.yml"

    kinematics = load_yaml(str(config_path))["robot_cfg"]["kinematics"]
    assets = get_assets_path()
    urdf_path = Path(join_path(assets, kinematics["urdf_path"]))
    asset_root = Path(join_path(assets, kinematics["asset_root_path"]))

    assert urdf_path.is_file()
    assert asset_root == assets / "robot/franka_description"
    root = ElementTree.parse(urdf_path).getroot()
    mesh_names = {
        mesh.attrib["filename"]
        for mesh in root.iter("mesh")
        if "filename" in mesh.attrib
    }
    assert mesh_names
    assert all((asset_root / mesh_name).is_file() for mesh_name in mesh_names)
    assert kinematics["base_link"] == "panda_link0"
    assert kinematics["tool_frames"] == ["panda_hand"]


def test_pinned_primitive_world_config_loads_without_accelerator_imports() -> None:
    world = load_yaml(join_path(get_scene_configs_path(), "collision_test.yml"))
    assert world["cuboid"]["table"]["dims"] == [1.4, 1.4, 0.05]
    assert world["cuboid"]["box"]["pose"] == [0.4, 0.0, 0.3, 1, 0, 0, 0.0]


def test_missing_robot_error_matches_pinned_inventory() -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"^Robot 'missing' not found\. Available robots: \['dual_ur10e', 'franka', 'simple_mimic_robot', 'unitree_g1', 'ur10e'\]$",
    ):
        get_robot_path("missing")


def test_load_yaml_keeps_native_missing_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError) as error:
        load_yaml(str(missing))
    assert error.value.filename == str(missing)
