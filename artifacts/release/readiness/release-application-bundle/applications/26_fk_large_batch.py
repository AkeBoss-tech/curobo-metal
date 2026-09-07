"""Run a larger FK batch and preserve input ordering."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("ur10e.yml"))
q = robot.default_joint_state.position.reshape(1, -1).repeat(64, 1)
q[:, 0] += torch.linspace(-0.2, 0.2, 64, device=q.device)
poses = robot.compute_kinematics(JointState.from_position(q, robot.joint_names)).tool_poses.position
assert poses.shape[0] == 64 and not torch.equal(poses[0], poses[-1])
emit({"poses": tensor(poses, values=False), "endpoints": tensor(poses[[0, -1]])})
