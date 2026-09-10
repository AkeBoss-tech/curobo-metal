"""Minimal installed-package smoke test using the public cuRobo namespace."""

import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg, JointState


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
robot = Kinematics(
    KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg(device))
)
state = robot.compute_kinematics(
    JointState.from_position(
        torch.zeros((1, robot.dof), device=device),
        joint_names=robot.joint_names,
    )
)

print(f"device={device.type}")
print(f"tool_position={state.tool_poses.position.reshape(-1, 3)[0].tolist()}")
print(f"robot_spheres={state.robot_spheres.shape[-2]}")
