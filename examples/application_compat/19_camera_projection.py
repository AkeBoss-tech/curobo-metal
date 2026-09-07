"""Generate calibrated camera rays through the observation API."""

import torch
from curobo.types import CameraObservation, DeviceCfg
from application_support import emit, tensor

cfg = DeviceCfg()
camera = CameraObservation(
    depth_image=cfg.to_device([[1.0, 1.0], [1.0, 1.0]]),
    intrinsics=cfg.to_device([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1.0]]),
    resolution=[2, 2], depth_to_meter=1.0,
)
camera.update_projection_rays()
assert camera.projection_rays.shape[-1] == 3
emit({"rays": tensor(camera.projection_rays)})
