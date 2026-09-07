"""Transform a point cloud into a pose and back."""

import torch
from curobo.types import DeviceCfg, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
pose = Pose.from_list([1, 2, 3, 1, 0, 0, 0], cfg)
points = cfg.to_device([[0, 0, 0], [1, -1, 2]])
world = pose.transform_points(points)
local = pose.inverse().transform_points(world)
assert torch.allclose(local, points, atol=1e-6)
emit({"world": tensor(world), "local": tensor(local)})
