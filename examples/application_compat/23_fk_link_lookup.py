"""Retrieve named link poses from a Franka kinematics result."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
state = JointState.from_position(robot.default_joint_state.position.reshape(1, -1), robot.joint_names)
result = robot.compute_kinematics(state)
frame = robot.tool_frames[-1]
pose = result.tool_poses.get_link_pose(frame)
assert torch.equal(pose.position, result.tool_poses.get_link_pose(frame).position)
emit({"position": tensor(pose.position), "quaternion": tensor(pose.quaternion), "frame": frame})
