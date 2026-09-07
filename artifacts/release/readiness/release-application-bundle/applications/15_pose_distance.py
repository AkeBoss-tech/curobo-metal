"""Measure public linear and angular pose distances."""

import torch
from curobo.types import DeviceCfg, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
a = Pose.from_list([0, 0, 0, 1, 0, 0, 0], cfg)
b = Pose.from_list([0.3, 0.4, 0, 0.9238795, 0, 0, 0.3826834], cfg)
linear = a.linear_distance(b)
angular = a.angular_distance(b)
assert torch.allclose(linear, linear.new_tensor([0.5]), atol=1e-6)
assert angular.item() > 0
emit({"linear": tensor(linear), "angular": tensor(angular)})
