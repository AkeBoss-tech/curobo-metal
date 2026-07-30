"""Portable subset of cuRoboV2 solver-core configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class SolverCoreCfg:
    robot_config: Any
    device_cfg: DeviceCfg = DeviceCfg()
    optimizer_configs: List[Any] = field(default_factory=list)
    optimizer_rollout_configs: List[Any] = field(default_factory=list)
    metrics_rollout_config: Any = None
    scene_collision_cfg: Any = None
    use_cuda_graph: bool = False
    random_seed: int = 123
    store_debug: bool = False

    def __post_init__(self):
        if self.use_cuda_graph:
            raise NotImplementedError(
                "CUDA Graph capture has no Metal equivalent; set use_cuda_graph=False"
            )


__all__ = ["SolverCoreCfg"]
