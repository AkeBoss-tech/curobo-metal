"""cuRoboV2-shaped robot loader configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.types import CSpaceParams, LinkParams
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class KinematicsLoaderCfg:
    base_link: str
    device_cfg: DeviceCfg = DeviceCfg()
    tool_frames: Optional[List[str]] = None
    collision_link_names: Optional[List[str]] = None
    collision_spheres: Union[None, str, Dict[str, Any]] = None
    collision_sphere_buffer: Union[float, Dict[str, float]] = 0.0
    self_collision_buffer: Optional[Dict[str, float]] = None
    self_collision_ignore: Optional[Dict[str, List[str]]] = None
    debug: Optional[Dict[str, Any]] = None
    asset_root_path: str = ""
    mesh_link_names: Optional[List[str]] = None
    grasp_contact_link_names: Optional[List[str]] = None
    load_tool_frames_with_mesh: bool = False
    urdf_path: Optional[str] = None
    lock_joints: Optional[Dict[str, float]] = None
    extra_links: Optional[Dict[str, LinkParams]] = None
    add_object_link: bool = False
    use_external_assets: bool = False
    external_asset_path: Optional[str] = None
    external_robot_configs_path: Optional[str] = None
    extra_collision_spheres: Optional[Dict[str, int]] = None
    load_collision_spheres: bool = True
    num_envs: int = 1
    cspace: Union[None, CSpaceParams, Dict[str, List[Any]]] = None
    load_meshes: bool = False
    use_global_cumul: bool = True
    format_version: float = 2.0

    def __post_init__(self) -> None:
        if not self.base_link:
            raise ValueError("base_link cannot be empty")
        if self.tool_frames is None:
            self.tool_frames = []
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.urdf_path is not None and self.urdf_path.lower().endswith((".usd", ".usda", ".usdc")):
            raise NotImplementedError("USD/Isaac assets are unsupported; provide URDF")
        if isinstance(self.cspace, dict):
            self.cspace = CSpaceParams(device_cfg=self.device_cfg, **self.cspace)


__all__ = ["KinematicsLoaderCfg"]
