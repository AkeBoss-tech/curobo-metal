"""Portable robot configuration builder.

The CUDA implementation uses ``trimesh`` and Warp to fit spheres to arbitrary
link meshes.  This version deliberately does not pretend that those kernels are
available.  It *does*, however, make the useful mesh-free subset work: URDF
links described by primitive spheres can be collected, edited, saved, and used
by the regular kinematics/collision stack on CPU or MPS.
"""

from __future__ import annotations

import time
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

try:  # Optional visual/mesh packages are not required for primitive spheres.
    import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = Any
try:
    import trimesh as _trimesh
except ModuleNotFoundError:
    _trimesh = Any
tqdm = _tqdm
trimesh = _trimesh

from curobo.content import get_assets_path, get_robot_configs_path
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit import SphereFitMetrics, SphereFitType
from curobo._src.geom.types import Sphere
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.loader import KinematicsLoaderCfg
from curobo._src.robot.parser import UrdfRobotParser
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.viser_visualizer import ViserVisualizer
from curobo._src.util.xrdf_util import convert_curobo_to_xrdf
from curobo._src.util_file import join_path, load_yaml, write_yaml
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo_metal.config.loaders import dump_yaml, load_robot_config


class _RobotBuilderPortableMixin:
    def __init__(
        self,
        urdf_path: str,
        asset_path: str = "",
        tool_frames: Optional[List[str]] = None,
        device_cfg: Optional[DeviceCfg] = None,
    ):
        self.device_cfg = DeviceCfg() if device_cfg is None else device_cfg
        self.urdf_path = str(Path(urdf_path).resolve())
        self.asset_path = str(Path(asset_path).resolve()) if asset_path else ""
        self._parser = UrdfRobotParser(self.urdf_path, mesh_root=self.asset_path)
        self._link_names = self._parser.get_link_names_from_urdf()
        self._joint_names = self._parser.get_joint_names_from_urdf()
        self._base_link = self._parser.root_link
        self._tool_frames = list(tool_frames or self._link_names[-1:])
        # ``get_link_geometry`` may raise for an external mesh.  Keep that
        # optional dependency boundary out of construction and discover it only
        # when the caller actually asks to fit it.
        self._mesh_link_names = list(self._link_names)
        self._collision_spheres: Optional[Dict[str, List[Dict]]] = None
        self._self_collision_ignore: Optional[Dict[str, List[str]]] = None
        self._self_collision_buffer: Dict[str, float] = {}
        self._cspace_config: Optional[Dict[str, Any]] = None
        self._link_metrics: Dict[str, SphereFitMetrics] = {}

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
        result._self_collision_ignore = {
            name: list(values) for name, values in robot.self_collision_ignore.items()
        }
        result._self_collision_buffer = dict(robot.self_collision_buffer)
        cspace = robot.to_mapping()["robot_cfg"]["kinematics"].get("cspace")
        result._cspace_config = None if not cspace else cspace
        return result

    @staticmethod
    def _resolve_clip_plane(axis: str, offset: float) -> tuple:
        if axis not in {"x", "y", "z", "-x", "-y", "-z"}:
            raise ValueError("axis must be x, y, z, -x, -y, or -z")
        vector = [0.0, 0.0, 0.0]
        vector["xyz".index(axis[-1])] = -1.0 if axis.startswith("-") else 1.0
        return tuple(vector), float(offset)

    def fit_collision_spheres(
        self,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_links: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> Dict[str, List[Dict]]:
        """Collect sphere primitives from every link into a collision model.

        Arbitrary triangle-mesh fitting requires the optional portable mesh
        fitting stack and is rejected precisely.  Sphere URDF geometry is an
        exact collision representation, so it is retained without approximation.
        """
        if sphere_density <= 0.0 or surface_radius < 0.0 or iterations < 1:
            raise ValueError("sphere_density and iterations must be positive; surface_radius cannot be negative")
        SphereFitType(fit_type)
        result: Dict[str, List[Dict]] = {}
        for link_name in self._mesh_link_names:
            clip_plane = None
            if clip_links and link_name in clip_links:
                axis, offset = clip_links[link_name]
                clip_plane = self._resolve_clip_plane(axis, offset)
            spheres = self._fit_single_link(
                link_name,
                sphere_density=sphere_density,
                surface_radius=surface_radius,
                fit_type=fit_type,
                use_collision_mesh=use_collision_mesh,
                iterations=iterations,
                coverage_weight=coverage_weight,
                protrusion_weight=protrusion_weight,
                compute_metrics=compute_metrics,
                clip_plane=clip_plane,
            )
            if spheres:
                result[link_name] = spheres
        self._collision_spheres = result
        return result

    def refit_link_spheres(
        self,
        link_name: str,
        num_spheres: Optional[int] = None,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_plane: Optional[tuple] = None,
    ) -> List[Dict]:
        """Refit one primitive-sphere link, retaining V2's call signature."""
        if self._collision_spheres is None:
            self._collision_spheres = {}
        spheres = self._fit_single_link(
            link_name,
            num_spheres=num_spheres,
            sphere_density=sphere_density,
            surface_radius=surface_radius,
            fit_type=fit_type,
            use_collision_mesh=use_collision_mesh,
            iterations=iterations,
            coverage_weight=coverage_weight,
            protrusion_weight=protrusion_weight,
            compute_metrics=compute_metrics,
            clip_plane=clip_plane,
        )
        if not spheres:
            raise ValueError(f"Link {link_name!r} has no portable sphere geometry")
        self._collision_spheres[link_name] = spheres
        return spheres

    def compute_collision_matrix(
        self,
        prune_collisions: bool = True,
        num_samples: int = 1000,
        batch_size: int = 10000,
        seed: int = 345,
        custom_ignore: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        if self._collision_spheres is None:
            raise ValueError("Must call fit_collision_spheres() before compute_collision_matrix()")
        if prune_collisions:
            # The portable primitive representation has no broad phase / sampled
            # pruning implementation.  Neighbor pairs are still safe ignores.
            log_warn(
                "Portable RobotBuilder does not prune sampled collision pairs; "
                "returning neighbouring-link ignores only."
            )
        del num_samples, batch_size, seed
        matrix = self._create_neighbor_ignore_matrix()
        if custom_ignore:
            self._self_collision_ignore = matrix
            self._merge_collision_ignore(custom_ignore)
            matrix = self._self_collision_ignore
        self._self_collision_ignore = matrix
        return matrix

    def add_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        if self._self_collision_ignore is None:
            self._self_collision_ignore = {}
        values = self._self_collision_ignore.setdefault(link_name, [])
        for name in ignore_links:
            if name not in values:
                values.append(name)

    def remove_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        if self._self_collision_ignore is None:
            return
        values = self._self_collision_ignore.get(link_name, [])
        self._self_collision_ignore[link_name] = [x for x in values if x not in ignore_links]

    def build(self) -> KinematicsLoaderCfg:
        if self._collision_spheres is None:
            log_warn("Building robot configuration without fitting spheres.")
            self._collision_spheres = {}
        if self._self_collision_ignore is None:
            log_warn("Building robot configuration without computing collision matrix.")
            self._self_collision_ignore = self._create_neighbor_ignore_matrix()
        cspace = deepcopy(self._cspace_config)
        if cspace is None:
            names = self._parser.get_actuated_joint_names()
            cspace = {
                "joint_names": names,
                "default_joint_position": [0.0] * len(names),
                "cspace_distance_weight": [1.0] * len(names),
                "null_space_weight": [1.0] * len(names),
                "max_acceleration": [10.0] * len(names),
                "max_jerk": [500.0] * len(names),
            }
        return KinematicsLoaderCfg(
            base_link=self._base_link,
            device_cfg=self.device_cfg,
            tool_frames=self._tool_frames,
            collision_link_names=self.collision_link_names,
            collision_spheres=self._collision_spheres,
            mesh_link_names=self.collision_link_names,
            self_collision_buffer=deepcopy(self._self_collision_buffer),
            self_collision_ignore=deepcopy(self._self_collision_ignore),
            asset_root_path=self.asset_path,
            urdf_path=self.urdf_path,
            cspace=cspace,
        )

    def save(
        self, config: KinematicsLoaderCfg, output_path: str, include_cspace: bool = True
    ) -> None:
        """Write a portable, reloadable cuRobo robot YAML.

        ``KinematicsLoaderCfg`` deliberately materializes ``cspace`` as
        ``CSpaceParams`` so the runtime has device tensors.  YAML cannot encode
        that object directly.  Serialize it back to lists here, retaining the
        configuration data rather than silently omitting cspace during a
        builder edit/save/reload workflow.
        """
        value = {"kinematics": self._config_mapping(
            config, include_cspace=include_cspace
        )}
        value["kinematics"]["format_version"] = 2.0
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dump_yaml(value), encoding="utf-8")

    def save_xrdf(
        self,
        config: KinematicsLoaderCfg,
        output_path: str,
        geometry_name: str = "collision_model",
    ) -> None:
        """Write the portable YAML-shaped XRDF collision representation.

        This covers spheres, collision buffers/ignores, tool frames, and cspace
        records.  It intentionally does not author mesh assets, USD stages, or
        Isaac metadata; those require their respective optional backends.
        """
        if not geometry_name or not isinstance(geometry_name, str):
            raise ValueError("geometry_name must be a non-empty string")
        kinematics = self._config_mapping(config, include_cspace=True)
        kinematics.setdefault("lock_joints", {})
        xrdf = convert_curobo_to_xrdf(
            {"robot_cfg": {"kinematics": kinematics}}, geometry_name=geometry_name
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(xrdf, str(output))

    def visualize(
        self,
        config: Optional[KinematicsLoaderCfg] = None,
        port: int = 8080,
        show_meshes: bool = False,
        show_spheres: bool = True,
        timeout_sec: int = -1,
    ):
        del config, port, show_meshes, show_spheres, timeout_sec
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
        return self._self_collision_ignore

    @property
    def num_spheres(self) -> int:
        return sum(len(value) for value in (self._collision_spheres or {}).values())

    @property
    def link_metrics(self) -> Dict[str, SphereFitMetrics]:
        return self._link_metrics

    def _fit_single_link(
        self,
        link_name: str,
        num_spheres: Optional[int] = None,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_plane: Optional[tuple] = None,
    ) -> Optional[List[Dict]]:
        """Fit portable collision spheres to one link's authored geometry."""
        if link_name not in self._link_names:
            raise ValueError(f"unknown robot link: {link_name}")
        geometry = self._parser.get_link_geometry(link_name, use_collision_mesh)
        # Lightweight wheels carry collision meshes but may omit large visual
        # DAE files.  Collision geometry remains the correct conservative
        # source for a sphere model when the requested visual mesh is absent.
        if not geometry and not use_collision_mesh:
            geometry = self._parser.get_link_geometry(link_name, True)
        spheres: List[Dict] = []
        for geometry_index, value in enumerate(geometry):
            if isinstance(value, Sphere) and (num_spheres is None or num_spheres == 1):
                spheres.append({
                    "center": list(value.position or value.pose[:3]),
                    "radius": float(value.radius),
                })
                if compute_metrics:
                    self._link_metrics[link_name] = SphereFitMetrics(
                        num_spheres=1, coverage=1.0, protrusion=0.0
                    )
                continue
            target = num_spheres
            if target is not None and len(geometry) > 1:
                remaining = max(1, int(target) - len(spheres))
                target = max(1, remaining // (len(geometry) - geometry_index))
            result = fit_spheres_to_mesh(
                value.get_mesh(process=False),
                num_spheres=target,
                sphere_density=sphere_density,
                surface_radius=surface_radius,
                fit_type=fit_type,
                iterations=iterations,
                coverage_weight=coverage_weight,
                protrusion_weight=protrusion_weight,
                compute_metrics=compute_metrics,
                clip_plane=clip_plane,
                device_cfg=self.device_cfg,
            )
            centers = result.centers
            if value.pose is not None:
                centers = Pose.from_list(value.pose, self.device_cfg).transform_points(
                    centers
                ).reshape(-1, 3)
            spheres.extend(
                {"center": center.detach().cpu().tolist(), "radius": float(radius.detach().cpu())}
                for center, radius in zip(centers, result.radii)
            )
            if compute_metrics and result.metrics is not None:
                self._link_metrics[link_name] = result.metrics
        if num_spheres is not None and len(spheres) > int(num_spheres):
            spheres = spheres[: int(num_spheres)]
            if link_name in self._link_metrics:
                self._link_metrics[link_name].num_spheres = len(spheres)
        return spheres or None

    def _create_neighbor_ignore_matrix(self) -> Dict[str, List[str]]:
        matrix = {name: [] for name in self._link_names}
        for child, parent in self._parser.link_parent.items():
            matrix.setdefault(child, []).append(parent)
            matrix.setdefault(parent, []).append(child)
        return matrix

    def _merge_collision_ignore(self, custom_ignore: Dict[str, List[str]]) -> None:
        for name, values in custom_ignore.items():
            self.add_collision_ignore(name, values)

    @staticmethod
    def _yaml_value(value: Any) -> Any:
        """Convert device/runtime values into conservative YAML primitives."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): RobotBuilder._yaml_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RobotBuilder._yaml_value(item) for item in value]
        if hasattr(value, "__dict__"):
            return {
                key: RobotBuilder._yaml_value(item)
                for key, item in vars(value).items()
                if key != "device_cfg"
            }
        return value

    @classmethod
    def _config_mapping(
        cls, config: KinematicsLoaderCfg, *, include_cspace: bool
    ) -> Dict[str, Any]:
        """Return exactly the serializable subset accepted by ``from_config``."""
        excluded = {"device_cfg", "load_collision_spheres", "num_envs"}
        return {
            key: cls._yaml_value(item)
            for key, item in vars(config).items()
            if key not in excluded
            and item is not None
            and (include_cspace or key != "cspace")
        }


class RobotBuilder(_RobotBuilderPortableMixin):
    """Pinned builder declaration delegating to the portable implementation."""

    def __init__(
        self,
        urdf_path: str,
        asset_path: str = "",
        tool_frames: Optional[List[str]] = None,
        device_cfg: Optional[DeviceCfg] = None,
    ):
        _RobotBuilderPortableMixin.__init__(self, urdf_path, asset_path, tool_frames, device_cfg)

    @classmethod
    def from_config(
        cls, config_path: str, device_cfg: Optional[DeviceCfg] = None
    ) -> "RobotBuilder":
        return _RobotBuilderPortableMixin.from_config.__func__(cls, config_path, device_cfg)

    def fit_collision_spheres(
        self,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_links: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> Dict[str, List[Dict]]:
        return _RobotBuilderPortableMixin.fit_collision_spheres(
            self, sphere_density, surface_radius, fit_type, use_collision_mesh,
            iterations, coverage_weight, protrusion_weight, compute_metrics, clip_links,
        )

    def refit_link_spheres(
        self,
        link_name: str,
        num_spheres: Optional[int] = None,
        sphere_density: float = 1.0,
        surface_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.MORPHIT,
        use_collision_mesh: bool = False,
        iterations: int = 200,
        coverage_weight: Optional[float] = None,
        protrusion_weight: Optional[float] = None,
        compute_metrics: bool = False,
        clip_plane: Optional[tuple] = None,
    ) -> List[Dict]:
        return _RobotBuilderPortableMixin.refit_link_spheres(
            self, link_name, num_spheres, sphere_density, surface_radius, fit_type,
            use_collision_mesh, iterations, coverage_weight, protrusion_weight,
            compute_metrics, clip_plane,
        )

    def compute_collision_matrix(
        self,
        prune_collisions: bool = True,
        num_samples: int = 1000,
        batch_size: int = 10000,
        seed: int = 345,
        custom_ignore: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        return _RobotBuilderPortableMixin.compute_collision_matrix(
            self, prune_collisions, num_samples, batch_size, seed, custom_ignore
        )

    def add_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        return _RobotBuilderPortableMixin.add_collision_ignore(self, link_name, ignore_links)

    def remove_collision_ignore(self, link_name: str, ignore_links: List[str]) -> None:
        return _RobotBuilderPortableMixin.remove_collision_ignore(self, link_name, ignore_links)

    def build(self) -> KinematicsLoaderCfg:
        return _RobotBuilderPortableMixin.build(self)

    def save(
        self, config: KinematicsLoaderCfg, output_path: str, include_cspace: bool = True
    ) -> None:
        return _RobotBuilderPortableMixin.save(self, config, output_path, include_cspace)

    def save_xrdf(
        self,
        config: KinematicsLoaderCfg,
        output_path: str,
        geometry_name: str = "collision_model",
    ) -> None:
        return _RobotBuilderPortableMixin.save_xrdf(self, config, output_path, geometry_name)

    def visualize(
        self,
        config: Optional[KinematicsLoaderCfg] = None,
        port: int = 8080,
        show_meshes: bool = False,
        show_spheres: bool = True,
        timeout_sec: int = -1,
    ) -> ViserVisualizer:
        return _RobotBuilderPortableMixin.visualize(
            self, config, port, show_meshes, show_spheres, timeout_sec
        )

    @property
    def tool_frames(self) -> List[str]:
        return _RobotBuilderPortableMixin.tool_frames.fget(self)

    @property
    def collision_link_names(self) -> List[str]:
        return _RobotBuilderPortableMixin.collision_link_names.fget(self)

    @property
    def collision_spheres(self) -> Optional[Dict[str, List[Dict]]]:
        return _RobotBuilderPortableMixin.collision_spheres.fget(self)

    @property
    def collision_matrix(self) -> Optional[Dict[str, List[str]]]:
        return _RobotBuilderPortableMixin.collision_matrix.fget(self)

    @property
    def num_spheres(self) -> int:
        return _RobotBuilderPortableMixin.num_spheres.fget(self)

    @property
    def link_metrics(self) -> Dict[str, SphereFitMetrics]:
        return _RobotBuilderPortableMixin.link_metrics.fget(self)


__all__ = ["RobotBuilder"]
