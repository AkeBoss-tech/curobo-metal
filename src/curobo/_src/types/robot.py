"""Robot configuration facade for the pinned internal import path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curobo_metal.config.robot import RobotCfg as _MetalRobotCfg

from .device_cfg import DeviceCfg


@dataclass
class RobotCfg:
    kinematics: Any
    dynamics: Any | None = None
    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def create(
        data: dict[str, Any] | "RobotCfg" | _MetalRobotCfg,
        device_cfg: DeviceCfg = DeviceCfg(),
        load_collision_spheres: bool = True,
        num_envs: int = 1,
    ) -> "RobotCfg":
        if isinstance(data, RobotCfg):
            return data
        metal = _MetalRobotCfg.create(
            data,
            device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres,
            num_envs=num_envs,
        )
        return RobotCfg(metal, dynamics=metal.dynamics, device_cfg=device_cfg)

    @staticmethod
    def from_basic(
        urdf_path: str,
        base_link: str,
        tool_frames: list[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_dynamics: bool = False,
    ) -> "RobotCfg":
        metal = _MetalRobotCfg.from_basic(
            urdf_path, base_link, tool_frames, device_cfg=device_cfg, load_dynamics=load_dynamics
        )
        return RobotCfg(metal, dynamics=metal.dynamics, device_cfg=device_cfg)

    @property
    def cspace(self) -> Any:
        return self.kinematics.cspace

    def write_config(self, file_path: str | Path) -> None:
        if not isinstance(self.kinematics, _MetalRobotCfg):
            raise NotImplementedError(
                "write_config requires a curobo-metal kinematics configuration"
            )
        self.kinematics.write_config(file_path)


__all__ = ["RobotCfg"]
