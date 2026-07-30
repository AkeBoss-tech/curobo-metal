from dataclasses import dataclass, field
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class DetectorCfg:
    n_mesh_points_coarse: int = 500
    n_observed_points_coarse: int = 2000
    n_rotation_samples: int = 64
    n_iterations_coarse: int = 50
    distance_threshold_coarse: float = 0.5
    n_mesh_points_fine: int = 2000
    n_observed_points_fine: int = 10000
    n_iterations_fine: int = 50
    distance_threshold_fine: float = 0.01
    use_svd: bool = False
    use_huber_loss: bool = True
    huber_delta: float = 0.02
    save_iterations: bool = False
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
