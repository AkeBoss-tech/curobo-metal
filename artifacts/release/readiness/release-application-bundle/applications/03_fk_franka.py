"""Batched Franka FK from packaged configuration with repeatability checks."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
assert robot.get_dof() == 7
q = robot.default_joint_state.position.reshape(1, -1).repeat(16, 1)
q[:, 0] += torch.linspace(-0.1, 0.1, 16, device=q.device)
state = JointState.from_position(q, robot.joint_names)
first = robot.compute_kinematics(state).tool_poses.position.clone()
second = robot.compute_kinematics(state).tool_poses.position
assert torch.allclose(first, second, atol=1e-6)
assert not torch.allclose(first[0], first[-1])
emit({"position": tensor(first), "joint_names": robot.joint_names})
