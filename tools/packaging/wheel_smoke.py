#!/usr/bin/env python3
"""Fail unless an installed curobo-metal wheel exposes its drop-in foundation."""

from __future__ import annotations

from pathlib import Path

import torch

import curobo
from curobo.content import get_assets_path, get_robot_path
from curobo.types import DeviceCfg, JointState, Pose
from curobo.util_file import load_yaml


def main() -> None:
    package_root = Path(curobo.__file__).resolve().parent
    assert "site-packages" in package_root.as_posix(), package_root

    robot_config = get_robot_path("franka")
    config = load_yaml(str(robot_config))["robot_cfg"]["kinematics"]
    urdf = get_assets_path() / config["urdf_path"]
    assert robot_config.is_file(), robot_config
    assert urdf.is_file(), urdf

    device_cfg = DeviceCfg(torch.device("cpu"), torch.float32)
    pose = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device_cfg)
    state = JointState.from_position(torch.zeros(1, 7))
    assert pose.get_matrix().shape == (1, 4, 4)
    assert state.position.shape == (1, 7)


if __name__ == "__main__":
    main()
