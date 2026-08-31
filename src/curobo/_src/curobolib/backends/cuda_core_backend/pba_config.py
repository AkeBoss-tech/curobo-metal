from pathlib import Path
from typing import List

from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class PBAKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("parallel_banding")
    def get_kernel_files(self) -> List[str]: return ["pba3d_kernel.cuh"]
    def get_include_dirs(self) -> List[Path]: return self.get_base_include_dirs() + [self.kernel_dir]


class PBALaunchCfg:
    @staticmethod
    def flood_z(sx: int, sy: int) -> LaunchConfig: return LaunchConfig((ceil_div(sx, 32), ceil_div(sy, 4)), (32, 4))
    @staticmethod
    def maurer_axis(sx: int, sz: int) -> LaunchConfig: return LaunchConfig((ceil_div(sx, 32), ceil_div(sz, 4)), (32, 4))
    @staticmethod
    def color_axis(sx: int, sz: int, m3: int = 2) -> LaunchConfig: return LaunchConfig((ceil_div(sx, 32), sz), (32, m3))
