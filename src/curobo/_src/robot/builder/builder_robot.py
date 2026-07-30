"""Portable robot configuration builder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from curobo._src.robot.loader import KinematicsLoaderCfg
from curobo._src.robot.parser import UrdfRobotParser
from curobo._src.types.device_cfg import DeviceCfg
from curobo_metal.config.loaders import dump_yaml, load_robot_config


class RobotBuilder:
    def __init__(
        self,
        urdf_path: str,
        asset_path: str = "",
        tool_frames: Optional[List[str]] = None,
        device_cfg: Optional[DeviceCfg] = None,
    ) -> None:
        self.urdf_path = str(Path(urdf_path).resolve())
        self.asset_path = asset_path
        self.device_cfg = DeviceCfg() if device_cfg is None else device_cfg
        self._parser = UrdfRobotParser(self.urdf_path, mesh_root=asset_path)
        self._tool_frames = list(tool_frames or [self._parser.get_link_names()[-1]])
        self._collision_spheres: Optional[Dict[str, List[Dict]]] = None
        self._collision_matrix: Optional[Dict[str, List[str]]] = None
        self._link_metrics: Dict[str, object] = {}

    @classmethod
    def from_config(
        cls, config_path: str, device_cfg: Optional[DeviceCfg] = None
    ) -> "RobotBuilder":
        robot = load_robot_config(config_path, device_cfg=device_cfg or DeviceCfg())
        result = cls(
            robot.urdf_path, tool_frames=robot.tool_frames,
            device_cfg=device_cfg,
        )
        spheres: Dict[str, List[Dict]] = {}
        for sphere in robot.collision_spheres:
            spheres.setdefault(sphere.link_name, []).append(
                {"center": list(sphere.center), "radius": sphere.radius}
            )
        result._collision_spheres = spheres
        result._collision_matrix = {
            name: list(values) for name, values in robot.self_collision_ignore.items()
        }
        return result

    @staticmethod
    def _resolve_clip_plane(axis: str, offset: float) -> tuple:
        if axis not in {"x", "y", "z", "-x", "-y", "-z"}:
            raise ValueError("axis must be x, y, z, -x, -y, or -z")
        vector = [0.0, 0.0, 0.0]
        vector["xyz".index(axis[-1])] = -1.0 if axis.startswith("-") else 1.0
        return tuple(vector), float(offset)

    def fit_collision_spheres(self, *args, **kwargs) -> Dict[str, List[Dict]]:
        del args, kwargs
        raise NotImplementedError(
            "mesh sphere fitting requires the optional trimesh/sphere-fit backend"
        )

    def refit_link_spheres(self, link_name: str, *args, **kwargs) -> List[Dict]:
        del link_name, args, kwargs
        raise NotImplementedError(
            "mesh sphere fitting requires the optional trimesh/sphere-fit backend"
        )

    def compute_collision_matrix(
        self,
        prune_collisions: bool = True,
        num_samples: int = 1000,
        batch_size: int = 10000,
        seed: int = 345,
        custom_ignore: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        del num_samples, batch_size, seed
        if prune_collisions:
            raise NotImplementedError(
                "sampled collision pruning requires a compiled self-collision model"
            )
        links = self._parser.get_link_names()
        matrix = {name: [] for name in links}
        for child, parent in self._parser.link_parent.items():
            matrix.setdefault(child, []).append(parent)
            matrix.setdefault(parent, []).append(child)
        if custom_ignore:
            self._collision_matrix = matrix
            self._merge_collision_ignore(custom_ignore)
            matrix = self._collision_matrix
        self._collision_matrix = matrix
        return matrix

    def add_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        if self._collision_matrix is None:
            self._collision_matrix = {}
        values = self._collision_matrix.setdefault(link_name, [])
        for name in ignore_links:
            if name not in values:
                values.append(name)

    def remove_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        if self._collision_matrix is None:
            return
        values = self._collision_matrix.get(link_name, [])
        self._collision_matrix[link_name] = [x for x in values if x not in ignore_links]

    def build(self) -> KinematicsLoaderCfg:
        return KinematicsLoaderCfg(
            base_link=self._parser.root_link,
            device_cfg=self.device_cfg,
            tool_frames=self._tool_frames,
            collision_link_names=self.collision_link_names,
            collision_spheres=self._collision_spheres,
            self_collision_ignore=self._collision_matrix,
            asset_root_path=self.asset_path,
            urdf_path=self.urdf_path,
        )

    def save(
        self, config: KinematicsLoaderCfg, output_path: str, include_cspace: bool = True
    ) -> None:
        value = {
            "robot_cfg": {"kinematics": {
                key: item for key, item in config.__dict__.items()
                if key != "device_cfg" and (include_cspace or key != "cspace")
            }}
        }
        Path(output_path).write_text(dump_yaml(value), encoding="utf-8")

    def save_xrdf(self, *args, **kwargs) -> None:
        del args, kwargs
        raise NotImplementedError("XRDF authoring is not implemented by the portable builder")

    def visualize(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("visualization requires the optional Viser backend")

    @property
    def tool_frames(self) -> List[str]:
        return self._tool_frames

    @property
    def collision_link_names(self) -> List[str]:
        return [] if self._collision_spheres is None else list(self._collision_spheres)

    @property
    def collision_spheres(self) -> Optional[Dict[str, List[Dict]]]:
        return self._collision_spheres

    @property
    def collision_matrix(self) -> Optional[Dict[str, List[str]]]:
        return self._collision_matrix

    @property
    def num_spheres(self) -> int:
        return sum(len(value) for value in (self._collision_spheres or {}).values())

    @property
    def link_metrics(self) -> Dict[str, object]:
        return self._link_metrics

    def _create_neighbor_ignore_matrix(self) -> Dict[str, List[str]]:
        return self.compute_collision_matrix(prune_collisions=False)

    def _merge_collision_ignore(self, custom_ignore: Dict[str, List[str]]) -> None:
        for name, values in custom_ignore.items():
            self.add_collision_ignore(name, values)


__all__ = ["RobotBuilder"]
