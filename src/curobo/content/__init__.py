# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paths and discovery helpers for the content bundled with cuRobo."""

from pathlib import Path
from typing import List

__all__ = [
    "get_content_root",
    "get_assets_path",
    "get_configs_path",
    "get_robot_configs_path",
    "get_task_configs_path",
    "get_scene_configs_path",
    "list_available_robots",
    "get_robot_path",
]


def get_content_root() -> Path:
    """Return the absolute root of the installed content package."""
    return Path(__file__).parent


def get_assets_path() -> Path:
    """Return the bundled robot and scene asset directory."""
    return get_content_root() / "assets"


def get_configs_path() -> Path:
    """Return the bundled configuration directory."""
    return get_content_root() / "configs"


def get_robot_configs_path() -> Path:
    """Return the bundled robot configuration directory."""
    return get_configs_path() / "robot"


def get_task_configs_path() -> Path:
    """Return the task configuration directory, whether or not this slice supplies it."""
    return get_configs_path() / "task"


def get_scene_configs_path() -> Path:
    """Return the bundled scene configuration directory."""
    return get_configs_path() / "scene"


def list_available_robots() -> List[str]:
    """Return sorted bundled YAML robot names without extensions."""
    robot_configs = get_robot_configs_path()
    if not robot_configs.exists():
        return []
    return sorted(f.stem for f in robot_configs.glob("*.y*ml"))


def get_robot_path(robot_name: str) -> Path:
    """Return a robot configuration path, matching pinned cuRoboV2 behavior."""
    robot_path = get_robot_configs_path() / (robot_name + ".yml")
    if not robot_path.exists():
        available = list_available_robots()
        raise FileNotFoundError(f"Robot '{robot_name}' not found. Available robots: {available}")
    return robot_path
