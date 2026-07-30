from dataclasses import dataclass
from typing import Optional
from curobo._src.types.pose import Pose


@dataclass
class DetectionResult:
    pose: Pose
    config: Optional[object]
    confidence: float
    alignment_error: float
    n_iterations: int
    compute_time: float = 0.0
