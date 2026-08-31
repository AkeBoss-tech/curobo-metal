"""Configuration loader for portable cuRobo kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.robot.types import CSpaceParams, JointLimits, SelfCollisionKinematicsCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import is_file_xrdf
from curobo._src.util.logging import log_and_raise
from curobo_metal.config.loaders import load_robot_config
from curobo._src.robot.parser import RobotParser
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.loader.kinematics_loader import KinematicsLoader
from curobo._src.robot.loader.util import load_robot_yaml


def _packaged_robot_file(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    packaged = Path(__file__).resolve().parents[3] / "content" / "configs" / "robot" / name
    return packaged if packaged.exists() else candidate


def _apply_locked_joints(robot: Any) -> None:
    locked = robot.metadata.get("lock_joints", {})
    if not locked:
        return
    fixed_values = {str(name): float(value) for name, value in locked.items()}
    locked_mimics: dict[str, tuple[str, float, float]] = {}
    for joint in robot.joints:
        if joint.mimic_joint in fixed_values:
            locked_mimics[joint.name] = (
                joint.mimic_joint, float(joint.mimic_multiplier), float(joint.mimic_offset)
            )
            fixed_values[joint.name] = (
                fixed_values[joint.mimic_joint] * float(joint.mimic_multiplier)
                + float(joint.mimic_offset)
            )
    robot.metadata["locked_mimic_joints"] = locked_mimics
    for joint in robot.joints:
        if joint.name not in fixed_values:
            continue
        value = fixed_values[joint.name]
        if joint.kind == "prismatic":
            joint.xyz = tuple(
                origin + value * axis for origin, axis in zip(joint.xyz, joint.axis)
            )
        elif joint.kind == "revolute" and value != 0.0:
            roll, pitch, yaw = joint.rpy
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            origin_rotation = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])
            axis = np.asarray(joint.axis, dtype=float)
            axis /= np.linalg.norm(axis)
            skew = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ])
            axis_rotation = (
                np.eye(3) + np.sin(value) * skew + (1.0 - np.cos(value)) * (skew @ skew)
            )
            rotation = origin_rotation @ axis_rotation
            new_pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
            if abs(abs(new_pitch) - np.pi / 2) < 1e-7:
                new_roll, new_yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1])), 0.0
            else:
                new_roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
                new_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            joint.rpy = (new_roll, new_pitch, new_yaw)
        joint.kind = "fixed"
        joint.mimic_joint = None
    original_names = list(robot.cspace.joint_names)
    names = [name for name in original_names if name not in locked]
    keep = [index for index, name in enumerate(original_names) if name not in locked]
    for field_name in (
        "default_joint_position",
        "max_velocity",
        "max_acceleration",
        "max_jerk",
        "cspace_distance_weight",
        "null_space_weight",
    ):
        values = getattr(robot.cspace, field_name, None)
        if isinstance(values, (list, tuple)) and len(values) == len(original_names):
            setattr(robot.cspace, field_name, [values[index] for index in keep])
    robot.cspace.joint_names = names


@dataclass
class _KinematicsCfgPortableMixin:
    device_cfg: DeviceCfg
    tool_frames: List[str]
    kinematics_config: KinematicsParams
    self_collision_config: Optional[Any] = None
    kinematics_parser: Optional[Any] = None
    generator_config: Optional[Any] = None

    def __post_init__(self) -> None:
        self.kinematics_config.make_contiguous()

    def get_joint_limits(self) -> Any:
        return self.kinematics_config.joint_limits

    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        tool_frames: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "KinematicsCfg":
        from curobo_metal.config.robot import RobotCfg
        robot = RobotCfg.from_basic(urdf_path, base_link, tool_frames, device_cfg)
        return KinematicsCfg(device_cfg, list(tool_frames), KinematicsParams(robot))

    @staticmethod
    def from_content_path(
        content_path: Any,
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs: Any,
    ) -> "KinematicsCfg":
        path = getattr(content_path, "robot_config_file", content_path)
        if path is None:
            raise ValueError("content_path must provide robot_config_file")
        return KinematicsCfg.from_robot_yaml_file(
            path, tool_frames=tool_frames, device_cfg=device_cfg, **kwargs
        )

    @staticmethod
    def from_robot_yaml_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "KinematicsCfg":
        if tool_frames is not None and not isinstance(tool_frames, list):
            raise ValueError(f"tool_frames must be a list, but is {type(tool_frames)}")
        source: Any = file_path
        if isinstance(file_path, str):
            source = _packaged_robot_file(file_path)
        if urdf_path is not None and isinstance(source, dict):
            source = dict(source)
            raw = source.setdefault("robot_cfg", {}).setdefault("kinematics", {})
            raw["urdf_path"] = urdf_path
        if kwargs and isinstance(source, dict):
            source = dict(source)
            source.setdefault("robot_cfg", {}).setdefault("kinematics", {}).update(kwargs)
        robot = load_robot_config(source, device_cfg=device_cfg)
        _apply_locked_joints(robot)
        if tool_frames is not None:
            robot.tool_frames = list(tool_frames)
        return KinematicsCfg(
            device_cfg, list(robot.tool_frames), KinematicsParams(robot),
            self_collision_config={
                "ignore": robot.self_collision_ignore,
                "buffer": robot.self_collision_buffer,
            },
        )

    @staticmethod
    def from_config_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
    ) -> "KinematicsCfg":
        return KinematicsCfg.from_robot_yaml_file(
            file_path, tool_frames, device_cfg, urdf_path
        )

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "KinematicsCfg":
        return KinematicsCfg.from_robot_yaml_file(data_dict, tool_frames, device_cfg)

    @staticmethod
    def from_config(config: Any) -> "KinematicsCfg":
        if isinstance(config, KinematicsCfg):
            return config
        # Preserve loader-config data rather than forcing consumers to first
        # instantiate the low-level loader themselves.
        if isinstance(config, KinematicsLoaderCfg):
            loader = KinematicsLoader(config)
            return KinematicsCfg(
                config.device_cfg,
                list(config.tool_frames or loader.kinematics_config.tool_frames),
                loader.kinematics_config,
                loader.self_collision_config,
                loader.kinematics_parser,
            )
        if isinstance(config, dict):
            return KinematicsCfg.from_data_dict(config)
        raise TypeError("config must be KinematicsCfg or a robot configuration mapping")

    @property
    def cspace(self) -> Any:
        return self.kinematics_config.cspace

    @property
    def collision_spheres(self) -> Any:
        """Expose the source collision geometry at the historical boundary."""
        return self.kinematics_config.robot_cfg.collision_spheres

    @property
    def dof(self) -> int:
        return self.kinematics_config.num_dof


@dataclass
class KinematicsCfg(_KinematicsCfgPortableMixin):
    """Pinned configuration declaration backed by the portable loader path."""

    device_cfg: DeviceCfg
    tool_frames: List[str]
    kinematics_config: KinematicsParams
    self_collision_config: Optional[SelfCollisionKinematicsCfg] = None
    kinematics_parser: Optional[RobotParser] = None
    generator_config: Optional[KinematicsLoaderCfg] = None

    def get_joint_limits(self) -> JointLimits:
        return _KinematicsCfgPortableMixin.get_joint_limits(self)

    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        tool_frames: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_basic_urdf(
            urdf_path, base_link, tool_frames, device_cfg
        )

    @staticmethod
    def from_content_path(
        content_path: ContentPath,
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs: Any,
    ) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_content_path(
            content_path, tool_frames, device_cfg, **kwargs
        )

    @staticmethod
    def from_robot_yaml_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
        **kwargs: Any,
    ) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_robot_yaml_file(
            file_path, tool_frames, device_cfg, urdf_path, **kwargs
        )

    @staticmethod
    def from_config_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
    ) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_config_file(
            file_path, tool_frames, device_cfg, urdf_path
        )

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_data_dict(
            data_dict, tool_frames, device_cfg
        )

    @staticmethod
    def from_config(config: KinematicsLoaderCfg) -> KinematicsCfg:
        return _KinematicsCfgPortableMixin.from_config(config)

    @property
    def cspace(self) -> CSpaceParams:
        return _KinematicsCfgPortableMixin.cspace.fget(self)

    @property
    def dof(self) -> int:
        return _KinematicsCfgPortableMixin.dof.fget(self)
