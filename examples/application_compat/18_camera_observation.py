"""Filter, clone, and validate a metric camera observation."""

import torch
from curobo.types import CameraObservation, DeviceCfg, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
depth = cfg.to_device([[0.005, 0.02], [0.03, 0.0]])
camera = CameraObservation(depth_image=depth, pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0], cfg))
camera.filter_depth(0.01)
clone = camera.clone()
clone.depth_image[0, 1] += 1
assert camera.depth_image[0, 0] == 0
assert not torch.equal(camera.depth_image, clone.depth_image)
emit({"filtered": tensor(camera.depth_image), "clone": tensor(clone.depth_image)})
