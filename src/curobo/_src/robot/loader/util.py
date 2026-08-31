"""Robot YAML/XRDF resolution."""

from curobo._src.types.content_path import ContentPath
from curobo._src.util.logging import log_and_raise
from curobo._src.util.xrdf_util import convert_xrdf_to_curobo
from curobo.util_file import join_path, load_yaml


def load_robot_yaml(content_path: ContentPath = ContentPath()) -> dict:
    if not isinstance(content_path, ContentPath):
        raise TypeError("content_path should be of type ContentPath")
    path = content_path.get_robot_configuration_path()
    robot_data = load_yaml(path)
    if robot_data.get("format") == "xrdf":
        robot_data = convert_xrdf_to_curobo(
            content_path=content_path, input_xrdf_dict=robot_data
        )
        robot_data["robot_cfg"]["kinematics"]["asset_root_path"] = (
            content_path.robot_asset_absolute_path
        )
    elif "robot_cfg" not in robot_data:
        robot_data = {"robot_cfg": {"kinematics": robot_data}}
    elif "kinematics" not in robot_data["robot_cfg"]:
        robot_data["robot_cfg"] = {"kinematics": robot_data["robot_cfg"]}
    kinematics = robot_data["robot_cfg"]["kinematics"]
    if content_path.robot_urdf_absolute_path is not None:
        kinematics["urdf_path"] = content_path.robot_urdf_absolute_path
    if content_path.robot_asset_absolute_path is not None:
        kinematics["asset_root_path"] = content_path.robot_asset_absolute_path
    if isinstance(kinematics.get("collision_spheres"), str):
        kinematics["collision_spheres"] = join_path(
            content_path.robot_config_root_path, kinematics["collision_spheres"]
        )
    return robot_data


__all__ = ["load_robot_yaml"]
