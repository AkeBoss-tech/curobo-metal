"""A second packaged robot and joint-name reordering through the standard API."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("ur10e.yml"))
assert robot.get_dof() == 6
q = robot.default_joint_state.position.reshape(1, -1).clone()
q[:, 0] += 0.1
state = JointState.from_position(q, robot.joint_names)
reordered = state.reorder(list(reversed(robot.joint_names))).reorder(robot.joint_names)
assert torch.equal(state.position, reordered.position)
pose = robot.compute_kinematics(reordered).tool_poses
emit({"position": tensor(pose.position), "quaternion": tensor(pose.quaternion), "joint_names": robot.joint_names})
