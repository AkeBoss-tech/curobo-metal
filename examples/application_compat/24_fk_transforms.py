"""Compute all Franka link transforms through the public robot model."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from application_support import emit, tensor

robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
transforms = robot.get_all_link_transforms().get_matrix()
assert transforms.shape[-2:] == (4, 4)
emit({"transforms": tensor(transforms, values=False),
      "last": tensor(transforms[..., -1, :, :])})
