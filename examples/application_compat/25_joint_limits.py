"""Read configured robot limits and verify the retract pose lies inside them."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
limits = robot.get_joint_limits()
q = robot.default_joint_state.position
assert torch.all(q >= limits.position[0]) and torch.all(q <= limits.position[1])
emit({"position": tensor(limits.position), "velocity": tensor(limits.velocity),
      "default": tensor(q)})
