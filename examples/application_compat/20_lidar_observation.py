"""Convert a calibrated planar LiDAR range image to points."""

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
points = lidar.to_pointcloud(project_to_pose=True)
assert points.shape == (1, 1, 4, 3)
assert lidar.valid_mask().all()
emit({"points": tensor(points), "valid": tensor(lidar.valid_mask())})
