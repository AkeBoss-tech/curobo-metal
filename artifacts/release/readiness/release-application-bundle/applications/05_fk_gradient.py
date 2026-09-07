"""Validate the public FK gradient against a central finite difference."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
q = robot.default_joint_state.position.reshape(1, -1).clone().requires_grad_()

def loss(position):
    pose = robot.compute_kinematics(JointState.from_position(position, robot.joint_names)).tool_poses
    return pose.position.square().sum()

loss(q).backward()
assert q.grad is not None and q.grad.abs().max().item() > 1e-5
step = torch.zeros_like(q)
step[:, 1] = 1e-3
with torch.no_grad():
    finite_difference = (loss(q + step) - loss(q - step)) / 2e-3
assert torch.allclose(q.grad[0, 1], finite_difference, atol=2e-3, rtol=2e-2)
emit({"gradient": tensor(q.grad), "finite_difference": tensor(finite_difference)})
