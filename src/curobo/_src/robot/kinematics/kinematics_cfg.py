"""Configuration loader for portable cuRobo kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo_metal.config.loaders import load_robot_config


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
    for joint in robot.joints:
        if joint.name not in locked:
            continue
        value = float(locked[joint.name])
        if joint.kind == "prismatic":
            joint.xyz = tuple(
                origin + value * axis for origin, axis in zip(joint.xyz, joint.axis)
            )
        elif joint.kind == "revolute" and value != 0.0:
            raise NotImplementedError(
                "locking a nonzero revolute joint requires transform composition"
            )
        joint.kind = "fixed"
    names = [name for name in robot.cspace.joint_names if name not in locked]
    positions = dict(zip(robot.cspace.joint_names, robot.cspace.default_joint_position))
    robot.cspace.joint_names = names
    robot.cspace.default_joint_position = [positions[name] for name in names]


@dataclass
class KinematicsCfg:
    device_cfg: DeviceCfg
    tool_frames: List[str]
    kinematics_config: KinematicsParams
    self_collision_config: Optional[Any] = None
    kinematics_parser: Optional[Any] = None
    generator_config: Optional[Any] = None

    def __post_init__(self) -> None:
        self.kinematics_config.make_contiguous()

    def get_joint_limits(self) -> Any:
        joints = {
            joint.name: joint.limits
            for joint in self.kinematics_config.robot_cfg.joints
            if joint.name in self.kinematics_config.joint_names
        }
        return joints

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
        if isinstance(config, dict):
            return KinematicsCfg.from_data_dict(config)
        raise TypeError("config must be KinematicsCfg or a robot configuration mapping")

    @property
    def cspace(self) -> Any:
        return self.kinematics_config.cspace

    @property
    def dof(self) -> int:
        return self.kinematics_config.num_dof
