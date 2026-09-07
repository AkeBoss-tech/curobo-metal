"""Clone and move a calibrated planar LiDAR range image."""

import torch
from curobo.types import DeviceCfg, LidarObservation, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
lidar = LidarObservation(
    range_image=cfg.to_device([[[1.0, 2.0, 1.0, 2.0]]]),
    pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0], cfg),
    valid_range_m=cfg.to_device([[0.1, 3.0]]),
    elevation_range_rad=cfg.to_device([[0.0, 0.0]]),
)
clone = lidar.clone().to(cfg.device)
assert clone.shape == (1, 1, 4)
assert clone.range_image is not lidar.range_image
assert torch.equal(clone.valid_range_m, lidar.valid_range_m)
emit({"ranges": tensor(clone.range_image), "bounds": tensor(clone.valid_range_m),
      "elevation": tensor(clone.elevation_range_rad)})
