"""Content roots and resolved configuration paths."""

from dataclasses import dataclass
from typing import Optional

from curobo.content import get_assets_path, get_robot_configs_path, get_scene_configs_path
from curobo.util_file import join_path


@dataclass(frozen=True)
class ContentPath:
    robot_config_root_path: str = get_robot_configs_path()
    robot_xrdf_root_path: str = get_robot_configs_path()
    robot_urdf_root_path: str = get_assets_path()
    robot_asset_root_path: str = get_assets_path()
    scene_config_root_path: str = get_scene_configs_path()
    world_asset_root_path: str = get_assets_path()
    robot_config_absolute_path: Optional[str] = None
    robot_xrdf_absolute_path: Optional[str] = None
    robot_urdf_absolute_path: Optional[str] = None
    robot_asset_absolute_path: Optional[str] = None
    scene_config_absolute_path: Optional[str] = None
    robot_config_file: Optional[str] = None
    robot_xrdf_file: Optional[str] = None
    robot_urdf_file: Optional[str] = None
    robot_asset_subroot_path: Optional[str] = None
    scene_config_file: Optional[str] = None

    def __post_init__(self) -> None:
        pairs = (
            ("robot_config_file", "robot_config_absolute_path", "robot_config_root_path"),
            ("robot_xrdf_file", "robot_xrdf_absolute_path", "robot_xrdf_root_path"),
            ("robot_urdf_file", "robot_urdf_absolute_path", "robot_urdf_root_path"),
            ("robot_asset_subroot_path", "robot_asset_absolute_path", "robot_asset_root_path"),
            ("scene_config_file", "scene_config_absolute_path", "scene_config_root_path"),
        )
        for relative, absolute, root in pairs:
            value = getattr(self, relative)
            if value is not None:
                if getattr(self, absolute) is not None:
                    raise ValueError(f"{relative} and {absolute} cannot be provided together.")
                object.__setattr__(self, absolute, join_path(getattr(self, root), value))

    def get_robot_configuration_path(self):
        if self.robot_config_absolute_path is not None:
            return self.robot_config_absolute_path
        if self.robot_xrdf_absolute_path is not None:
            return self.robot_xrdf_absolute_path
        raise ValueError("No Robot configuration file found")


__all__ = ["ContentPath"]
