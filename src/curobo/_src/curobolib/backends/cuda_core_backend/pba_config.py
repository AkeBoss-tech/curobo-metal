from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class PBAKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("parallel_banding")
    def get_kernel_files(self): return ["pba3d_kernel.cuh"]
    def get_include_dirs(self): return self.get_base_include_dirs() + [self.kernel_dir]


class PBALaunchCfg:
    @staticmethod
    def flood_z(sx, sy): return LaunchConfig((ceil_div(sx, 32), ceil_div(sy, 4)), (32, 4))
    @staticmethod
    def maurer_axis(sx, sz): return LaunchConfig((ceil_div(sx, 32), ceil_div(sz, 4)), (32, 4))
    @staticmethod
    def color_axis(sx, sz, m3=2): return LaunchConfig((ceil_div(sx, 32), sz), (32, m3))
