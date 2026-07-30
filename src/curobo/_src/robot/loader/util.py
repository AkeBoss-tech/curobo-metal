"""Robot YAML/XRDF resolution."""

from curobo._src.types.content_path import ContentPath
from curobo.util_file import load_yaml


def load_robot_yaml(content_path: ContentPath = ContentPath()) -> dict:
    path = content_path.get_robot_configuration_path()
    if str(path).lower().endswith(".xrdf"):
        raise NotImplementedError(
            "standalone XRDF conversion requires an accompanying URDF; use RobotCfg.create"
        )
    return load_yaml(path)


__all__ = ["load_robot_yaml"]
