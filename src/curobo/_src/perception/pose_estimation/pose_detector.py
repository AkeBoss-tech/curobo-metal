import time
import torch
from curobo._src.types.pose import Pose
from .detection_result import DetectionResult


class PoseDetector:
    def __init__(self, config, mesh):
        self.config, self.mesh = config, mesh

    def detect(self, pointcloud, *args, **kwargs):
        start = time.perf_counter()
        points = torch.as_tensor(pointcloud, device=self.config.device_cfg.device, dtype=self.config.device_cfg.dtype)
        model = self.mesh.vertices.to(points)
        translation = points.reshape(-1,3).mean(0)-model.reshape(-1,3).mean(0)
        pose = Pose(translation[None], translation.new_tensor([[1.,0.,0.,0.]]))
        shifted = model+translation
        error = torch.cdist(points.reshape(-1,3), shifted.reshape(-1,3)).min(-1).values.mean()
        return DetectionResult(pose, self.config, float(torch.exp(-error)), float(error), 1, time.perf_counter()-start)
