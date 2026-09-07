"""Pose composition, inverse, quaternion convention, and clone ownership."""

import torch
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import Pose
from application_support import emit, tensor

# Construct the robot context first, as in ordinary planning applications.
# Upstream initializes its geometry runtime during this public API setup.
robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))

left = Pose.from_list([1, 0, 0, 1, 0, 0, 0])
right = Pose.from_list([0, 2, 0, 0, 0, 0, 1], q_xyzw=True)
composed = left.multiply(right)
assert torch.allclose(composed.position, composed.position.new_tensor([[1, 2, 0]]))
identity = composed.inverse().multiply(composed).get_matrix()
assert torch.allclose(identity, torch.eye(4, device=identity.device).expand_as(identity), atol=1e-6)
clone = composed.clone()
clone.position.add_(1)
assert not torch.equal(clone.position, composed.position)
emit({"matrix": tensor(composed.get_matrix()), "identity": tensor(identity)})
