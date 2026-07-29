"""Narrow, dependency-free compatibility surface for pinned cuRoboV2 data."""

from .curobo_v2 import (
    CUROBO_V2_REVISION,
    BackendRobotConfig,
    UnsupportedCuroboConfig,
    convert_kinematics_config,
)

__all__ = [
    "CUROBO_V2_REVISION",
    "BackendRobotConfig",
    "UnsupportedCuroboConfig",
    "convert_kinematics_config",
]
