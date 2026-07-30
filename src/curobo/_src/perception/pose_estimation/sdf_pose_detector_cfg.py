from dataclasses import dataclass, field
from .pose_detector_cfg import DetectorCfg


@dataclass
class SDFDetectorCfg(DetectorCfg):
    surface_threshold: float = 0.01
