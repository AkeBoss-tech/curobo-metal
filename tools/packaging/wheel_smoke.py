#!/usr/bin/env python3
"""Fail unless an installed curobo-metal wheel exposes its drop-in foundation."""

from __future__ import annotations

from pathlib import Path

import torch

import curobo
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.content import get_assets_path, get_robot_path
from curobo.kinematics import Kinematics, KinematicsCfg
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

    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=device_cfg
    )
    kinematics = Kinematics(kinematics_cfg)
    robot_state = kinematics.compute_kinematics(
        JointState.from_position(
            kinematics.default_joint_position,
            joint_names=kinematics.joint_names,
        )
    )
    assert robot_state.tool_poses.position.shape == (1, 1, 1, 3)
    assert robot_state.robot_spheres.shape[-2:] == (61, 4)

    # These imports are part of the public drop-in contract. Construction is
    # covered by the source suite because it requires a complete scene config.
    assert RobotCollisionChecker.__name__ == "RobotSceneCollision"
    assert RobotCollisionCheckerCfg.__name__ == "RobotSceneCollisionCfg"


if __name__ == "__main__":
    main()
