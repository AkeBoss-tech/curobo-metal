"""Configuration for the portable SDF pose-refinement facade.

The field names intentionally follow the pinned V2 configuration.  CUDA graph
capture is a no-op hint on CPU/MPS; the actual optimizer is an eager,
deterministic tensor ICP/SDF approximation.
"""

from dataclasses import dataclass, field

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class SDFDetectorCfg:
    max_iterations: int = 100
    inner_iterations: int = 25
    convergence_threshold: float = 1e-5
    rotation_convergence_threshold: float = 1e-5
    use_cuda_graph: bool = True
    distance_threshold: float = 0.2
    min_valid_ratio: float = 0.1
    use_huber: bool = True
    huber_delta: float = 0.1
    lambda_initial: float = 1e-3
    lambda_factor: float = 10.0
    lambda_min: float = 1e-7
    lambda_max: float = 1e7
    rho_min: float = 0.25
    n_points: int = 5000
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    def __post_init__(self) -> None:
        if self.max_iterations < 0 or self.inner_iterations < 1:
            raise ValueError("max_iterations must be nonnegative and inner_iterations positive")
        if self.n_points < 1:
            raise ValueError("n_points must be positive")
        if self.distance_threshold <= 0 or not 0 < self.min_valid_ratio <= 1:
            raise ValueError("distance_threshold and min_valid_ratio must be positive")
        if self.huber_delta <= 0 or self.lambda_initial <= 0:
            raise ValueError("huber_delta and lambda_initial must be positive")

    @property
    def max_distance(self):
        """Pinned alias used by mesh-SDF callers."""
        return self.distance_threshold
