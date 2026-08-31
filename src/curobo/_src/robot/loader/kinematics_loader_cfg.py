"""cuRoboV2-shaped robot loader configuration."""

from __future__ import annotations

import copy
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.types import CSpaceParams, LinkParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo.content import get_assets_path, get_robot_configs_path
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import join_path
from curobo.util_file import load_yaml


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
        """Normalize a portable robot-loader description without CUDA imports.

        The upstream object accepts configuration-relative asset paths and a
        mixture of YAML records and value objects.  Keeping this conversion at
        the boundary gives :class:`KinematicsLoader` an immutable-ish,
        device-aware description to compile on either CPU or MPS.  USD/Isaac
        and external CUDA asset providers deliberately remain explicit
        boundaries.
        """
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not self.base_link:
            raise ValueError("base_link cannot be empty")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.urdf_path is not None and self.urdf_path.lower().endswith((".usd", ".usda", ".usdc")):
            raise NotImplementedError("USD/Isaac assets are unsupported; provide URDF")
        if self.use_external_assets:
            raise NotImplementedError(
                "external CUDA/Isaac asset providers are unsupported; pass a local URDF path"
            )

        self.urdf_path = self._resolve_asset_path(self.urdf_path, "URDF")
        self.asset_root_path = self._resolve_asset_root(self.asset_root_path)
        if not self.tool_frames:
            raise ValueError("tool_frames must specify at least one end-effector link")
        self.tool_frames = self._unique_names(self.tool_frames, "tool_frames")
        self.collision_link_names = self._unique_names(
            self.collision_link_names or [], "collision_link_names"
        )
        self.mesh_link_names = self._unique_names(self.mesh_link_names or [], "mesh_link_names")
        self.grasp_contact_link_names = None if self.grasp_contact_link_names is None else self._unique_names(
            self.grasp_contact_link_names, "grasp_contact_link_names"
        )
        if self.load_tool_frames_with_mesh:
            self.tool_frames = self._unique_names(
                [*self.tool_frames, *self.mesh_link_names], "tool_frames"
            )

        self.lock_joints = self._float_mapping(self.lock_joints, "lock_joints")
        self.self_collision_buffer = self._float_mapping(
            self.self_collision_buffer, "self_collision_buffer"
        )
        self.self_collision_ignore = self._name_list_mapping(
            self.self_collision_ignore, "self_collision_ignore"
        )
        self.extra_links = self._normalise_extra_links(self.extra_links)

        if not self.load_collision_spheres:
            self.collision_spheres = None
            self.collision_link_names = []
            self.extra_collision_spheres = None
        else:
            self.collision_spheres = self._normalise_collision_spheres(self.collision_spheres)
            if self.collision_spheres is None and self.collision_link_names:
                raise ValueError(
                    "collision_link_names are provided without collision_spheres"
                )
            self._append_extra_collision_spheres()

        if isinstance(self.cspace, dict):
            data = deepcopy(self.cspace)
            data.setdefault("device_cfg", self.device_cfg)
            self.cspace = CSpaceParams(**data)
        if self.cspace is not None and not isinstance(self.cspace, CSpaceParams):
            raise TypeError("cspace must be CSpaceParams, a mapping, or None")

    @staticmethod
    def _unique_names(values: List[str], field: str) -> List[str]:
        result = list(values)
        if any(not isinstance(value, str) or not value for value in result):
            raise TypeError(f"{field} must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"{field} must not contain duplicate names")
        return result

    @staticmethod
    def _float_mapping(
        value: Optional[Dict[str, float]], field: str
    ) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{field} must be a mapping or None")
        result: Dict[str, float] = {}
        for name, amount in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"{field} keys must be non-empty strings")
            numeric = float(amount)
            if numeric != numeric or numeric in (float("inf"), -float("inf")):
                raise ValueError(f"{field} values must be finite")
            result[name] = numeric
        return result

    @classmethod
    def _name_list_mapping(
        cls, value: Optional[Dict[str, List[str]]], field: str
    ) -> Optional[Dict[str, List[str]]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"{field} must be a mapping or None")
        result: Dict[str, List[str]] = {}
        for name, names in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"{field} keys must be non-empty strings")
            result[name] = cls._unique_names(list(names), f"{field}[{name!r}]")
        return result

    @staticmethod
    def _resolve_asset_path(path: Optional[str], label: str) -> Optional[str]:
        if path is None:
            return None
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        packaged = get_assets_path() / candidate
        if packaged.exists():
            return str(packaged.resolve())
        # Keep an unresolved relative path as a useful file-not-found error
        # from the parser rather than silently rewriting it to an arbitrary
        # working directory.
        return str(candidate)

    @staticmethod
    def _resolve_asset_root(path: str) -> str:
        if not path:
            return ""
        candidate = Path(path).expanduser()
        return str(candidate if candidate.is_absolute() else (get_assets_path() / candidate).resolve())

    @staticmethod
    def _normalise_extra_links(
        value: Optional[Dict[str, LinkParams]]
    ) -> Dict[str, LinkParams]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("extra_links must be a mapping or None")
        result: Dict[str, LinkParams] = {}
        for name, params in value.items():
            item = LinkParams.create(params) if isinstance(params, dict) else params
            if not isinstance(item, LinkParams):
                raise TypeError("extra_links values must be LinkParams or mappings")
            if name != item.link_name:
                raise ValueError("extra_links key must match LinkParams.link_name")
            result[name] = item
        return result

    @staticmethod
    def _normalise_collision_spheres(
        value: Union[None, str, Dict[str, Any]]
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if value is None:
            return None
        if isinstance(value, str):
            source = Path(value)
            if not source.is_absolute():
                source = get_robot_configs_path() / source
            payload = load_yaml(str(source))
            value = payload.get("collision_spheres", payload)
        if not isinstance(value, dict):
            raise TypeError("collision_spheres must be a mapping, YAML path, or None")
        result: Dict[str, List[Dict[str, Any]]] = {}
        for link_name, rows in value.items():
            if not isinstance(link_name, str) or not link_name or not isinstance(rows, list):
                raise TypeError("collision_spheres must map link names to sphere lists")
            result[link_name] = []
            for row in rows:
                if not isinstance(row, dict) or "center" not in row or "radius" not in row:
                    raise ValueError("each collision sphere requires center and radius")
                center = [float(item) for item in row["center"]]
                radius = float(row["radius"])
                if len(center) != 3:
                    raise ValueError("collision sphere center must contain three values")
                if any(item != item or item in (float("inf"), -float("inf")) for item in center):
                    raise ValueError("collision sphere centers must be finite")
                if radius != radius or radius in (float("inf"), -float("inf")):
                    raise ValueError("collision sphere radius must be finite")
                result[link_name].append({"center": center, "radius": radius})
        return result

    def _append_extra_collision_spheres(self) -> None:
        if self.extra_collision_spheres is None:
            return
        if self.collision_spheres is None:
            raise ValueError("extra_collision_spheres requires collision_spheres")
        for name, count in self.extra_collision_spheres.items():
            if not isinstance(name, str) or not name or isinstance(count, bool) or int(count) != count:
                raise TypeError("extra_collision_spheres must map link names to integer counts")
            if count < 0:
                raise ValueError("extra_collision_spheres counts must be non-negative")
            self.collision_spheres.setdefault(name, []).extend(
                {"center": [0.0, 0.0, 0.0], "radius": -100.0}
                for _ in range(int(count))
            )


__all__ = ["KinematicsLoaderCfg"]
