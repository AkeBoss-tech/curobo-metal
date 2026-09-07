"""Portable :class:`RobotCfg` value lifecycle for the pinned V2 import path.

The upstream record is deliberately small: it owns a compiled kinematics
configuration, optionally owns its dynamics configuration, and is the input
accepted by every high-level solver factory.  The CUDA implementation creates
the first of those through a loader.  This module keeps the same ownership
model while accepting the portable tree model as the compiled representation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from curobo_metal.config.robot import RobotCfg as _MetalRobotCfg

from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.util_file import write_yaml

from .device_cfg import DeviceCfg


@dataclass
class _PortableRobotCfg:
    """Robot configuration consumed by portable CPU/MPS solvers.

    ``kinematics`` is normally a portable
    :class:`curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg`, but
    retaining a raw ``curobo_metal`` tree is important for the long-standing
    direct construction pattern ``RobotCfg(params.robot_cfg)``.  Both forms
    expose ``cspace`` and can be converted to the typed dynamics configuration
    on demand.  CUDA/Isaac robot-model objects are intentionally not accepted.
    """

    kinematics: Any
    dynamics: Any | None = None
    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self) -> None:
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if self.kinematics is None:
            raise TypeError("kinematics must be a portable kinematics configuration")
        if not hasattr(self.kinematics, "cspace"):
            raise TypeError("kinematics must expose cspace")

    @staticmethod
    def _extract_kinematics(value: Mapping[str, Any]) -> tuple[Any, bool]:
        """Read a V2-shaped mapping without mutating the caller's objects."""
        raw: Any = value.get("robot_cfg", value)
        if not isinstance(raw, Mapping):
            raise TypeError("robot configuration must be a mapping")
        if "kinematics" not in raw:
            raise ValueError("robot configuration requires a kinematics entry")
        return raw["kinematics"], bool(raw.get("load_dynamics", False))

    @staticmethod
    def _kinematics_params(kinematics: Any) -> Any:
        """Return the portable typed parameter record for a dynamics request."""
        from curobo._src.robot.types import KinematicsParams

        if isinstance(kinematics, KinematicsParams):
            return kinematics
        params = getattr(kinematics, "kinematics_config", None)
        if isinstance(params, KinematicsParams):
            return params
        # Raw portable tree models are intentionally accepted by direct
        # construction.  KinematicsParams validates their topology and keeps
        # the normal tensor/device semantics used by DynamicsCfg.
        return KinematicsParams(kinematics)

    @staticmethod
    def _create_dynamics_config(
        kinematics_config: Any,
        device_cfg: DeviceCfg,
    ) -> Any:
        """Create the public typed dynamics record from portable kinematics.

        The raw CUDA packed dynamics config has no meaningful Metal analogue;
        ``DynamicsCfg`` is the portable production representation and is
        consumed by :class:`curobo._src.robot.dynamics.Dynamics`.
        """
        from curobo._src.robot.dynamics import DynamicsCfg
        from curobo._src.robot.types import KinematicsParams

        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(kinematics_config, KinematicsParams):
            raise TypeError("kinematics_config must be KinematicsParams")
        if kinematics_config.device_cfg != device_cfg:
            raise ValueError("kinematics_config.device_cfg must match device_cfg")
        return DynamicsCfg(kinematics_config=kinematics_config, device_cfg=device_cfg)

    @classmethod
    def create(
        cls,
        data: Mapping[str, Any] | "RobotCfg" | _MetalRobotCfg | str | PathLike[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_collision_spheres: bool = True,
        num_envs: int = 1,
    ) -> "RobotCfg":
        """Create a typed robot config from V2 YAML/mapping input.

        Existing ``RobotCfg`` values retain identity exactly as upstream.
        Mappings carrying an already-compiled ``KinematicsCfg`` also work,
        which is useful to callers that have loaded and reduced a robot before
        assembling solver settings.  Files/mappings otherwise route to the
        production portable loader.
        """
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(load_collision_spheres, bool):
            raise TypeError("load_collision_spheres must be a bool")
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(data, cls):
            return data

        if isinstance(data, Mapping):
            # The upstream factory accepts a precompiled KinematicsCfg under
            # ``kinematics``.  Do this before routing mappings to the YAML
            # loader, which necessarily expects a serializable mapping.
            source = deepcopy(dict(data))
            candidate, load_dynamics = cls._extract_kinematics(source)
            if not isinstance(candidate, Mapping):
                params = cls._kinematics_params(candidate)
                if params.device_cfg != device_cfg:
                    raise ValueError("kinematics.device_cfg must match device_cfg")
                dynamics = (
                    cls._create_dynamics_config(params, device_cfg)
                    if load_dynamics
                    else None
                )
                return cls(candidate, dynamics=dynamics, device_cfg=device_cfg)
            payload: Any = source
            request_dynamics = load_dynamics
        elif isinstance(data, _MetalRobotCfg):
            # The direct portable-tree form is common in lower-level V2
            # factories.  Do not force a YAML round trip or mutate the model.
            return cls(data, dynamics=getattr(data, "dynamics", None), device_cfg=device_cfg)
        elif isinstance(data, (str, Path, PathLike)):
            payload = data
            request_dynamics = False
        else:
            raise TypeError("robot config must be a mapping, path, RobotCfg, or portable robot model")

        metal = _MetalRobotCfg.create(
            payload,
            device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres,
            num_envs=num_envs,
        )
        from curobo._src.robot.kinematics.kinematics_cfg import (
            KinematicsCfg,
            _apply_locked_joints,
            _self_collision_from_robot,
        )

        _apply_locked_joints(metal)
        params = cls._kinematics_params(metal)
        if num_envs != params.num_envs:
            params.set_num_envs(num_envs)
        kinematics = KinematicsCfg(
            device_cfg,
            list(metal.tool_frames),
            params,
            self_collision_config=_self_collision_from_robot(metal, params, device_cfg),
        )
        dynamics = (
            cls._create_dynamics_config(params, device_cfg)
            if request_dynamics
            else None
        )
        return cls(kinematics, dynamics=dynamics, device_cfg=device_cfg)

    @classmethod
    def from_basic(
        cls,
        urdf_path: str | PathLike[str],
        base_link: str,
        tool_frames: list[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_dynamics: bool = False,
    ) -> "RobotCfg":
        """Build a portable tree from basic URDF inputs."""
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(base_link, str) or not base_link:
            raise ValueError("base_link must be a non-empty string")
        if not isinstance(tool_frames, list) or not tool_frames or any(
            not isinstance(name, str) or not name for name in tool_frames
        ):
            raise ValueError("tool_frames must be a non-empty list of strings")
        if not isinstance(load_dynamics, bool):
            raise TypeError("load_dynamics must be a bool")
        metal = _MetalRobotCfg.from_basic(
            urdf_path, base_link, tool_frames, device_cfg=device_cfg, load_dynamics=False
        )
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

        params = cls._kinematics_params(metal)
        kinematics = KinematicsCfg(device_cfg, list(tool_frames), params)
        dynamics = (
            cls._create_dynamics_config(params, device_cfg)
            if load_dynamics
            else None
        )
        return cls(kinematics, dynamics=dynamics, device_cfg=device_cfg)

    @property
    def cspace(self) -> Any:
        """Configuration-space record owned by this robot's kinematics."""
        return self.kinematics.cspace

    def clone(self) -> "RobotCfg":
        """Return an independent portable value copy.

        This is additive to the upstream dataclass surface, but prevents
        mutable c-space/robot records from being accidentally shared by
        CPU/MPS solver configurations.
        """
        return type(self)(
            deepcopy(self.kinematics), deepcopy(self.dynamics), self.device_cfg.clone()
        )

    copy = clone

    def write_config(self, file_path: str | Path) -> None:
        """Serialize the owned portable robot model to YAML.

        Typed ``KinematicsCfg`` wraps its original tree in
        ``kinematics_config.robot_cfg``; raw tree construction stores the
        model directly.  CUDA/Isaac-only configurations cannot be serialized
        through this adapter and fail rather than producing a misleading file.
        """
        model = self.kinematics
        if not hasattr(model, "write_config"):
            params = getattr(model, "kinematics_config", None)
            model = getattr(params, "robot_cfg", None)
        if not isinstance(model, _MetalRobotCfg):
            raise NotImplementedError(
                "write_config requires a portable curobo-metal robot configuration"
            )
        model.write_config(file_path)


@dataclass
class RobotCfg(_PortableRobotCfg):
    """Pinned RobotCfg surface backed by the richer portable implementation."""

    kinematics: KinematicsCfg
    dynamics: Optional[DynamicsCfg] = None
    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def _create_dynamics_config(
        kinematics_config,
        device_cfg: DeviceCfg,
    ) -> Optional[DynamicsCfg]:
        return _PortableRobotCfg._create_dynamics_config(kinematics_config, device_cfg)

    @staticmethod
    def create(
        data: Union[Dict[str, Any], "RobotCfg"],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_collision_spheres: bool = True,
        num_envs: int = 1,
    ) -> "RobotCfg":
        if isinstance(data, RobotCfg):
            return data
        value = _PortableRobotCfg.create.__func__(
            RobotCfg,
            data,
            device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres,
            num_envs=num_envs,
        )
        return value

    @staticmethod
    def from_basic(
        urdf_path: str,
        base_link: str,
        tool_frames: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_dynamics: bool = False,
    ):
        return _PortableRobotCfg.from_basic.__func__(
            RobotCfg,
            urdf_path,
            base_link,
            tool_frames,
            device_cfg=device_cfg,
            load_dynamics=load_dynamics,
        )

    def write_config(self, file_path):
        return super().write_config(file_path)

    @property
    def cspace(self) -> CSpaceParams:
        return super().cspace


__all__ = ["RobotCfg"]
