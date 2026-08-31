from pathlib import Path
from typing import List, Tuple

from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg


class OptimizationKernelCfg(CudaCoreKernelCfg):
    def __init__(self):
        super().__init__("optimization")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        return {"line_search": ["line_search/line_search_kernel.cuh"], "lbfgs": ["lbfgs/lbfgs_step_kernel.cuh"]}.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        return self.get_base_include_dirs() + [
            self.kernel_dir,
            self.kernel_dir / "line_search",
            self.kernel_dir / "lbfgs",
        ]


class LineSearchLaunchCfg:
    @staticmethod
    def calculate_config(opt_dim: int, batchsize: int) -> LaunchConfig:
        return LaunchConfig(batchsize, opt_dim, 0)


class LBFGSLaunchCfg:
    @staticmethod
    def calculate_config(
        batch_size: int, v_dim: int, history_m: int, use_shared_buffers: bool
    ) -> Tuple[LaunchConfig, bool, int]:
        basic = history_m * 4
        requested = (((2 * v_dim) + 2) * history_m + 33) * 4
        actual = bool(use_shared_buffers and requested <= 65536)
        shmem = requested if use_shared_buffers else basic
        return LaunchConfig(batch_size, v_dim, shmem), actual, requested if actual else 48000
