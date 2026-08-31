from pathlib import Path
from typing import List, Tuple

from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class KinematicsKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("kinematics")
    def get_kernel_files(self, kernel_type: str) -> List[str]:
        return {"forward": ["kinematics_forward_kernel.cuh"], "backward": ["kinematics_backward_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self) -> List[Path]: return self.get_base_include_dirs() + [self.kernel_dir]


class KinematicsLaunchCfg:
    MAX_FW_BATCH_PER_BLOCK = 8
    MAX_BW_BATCH_PER_BLOCK = 32
    DEFAULT_MAX_THREADS = 128
    DEFAULT_MAX_SHARED_MEM = 48 * 1024
    MAX_FORWARD_LINKS = 500

    @staticmethod
    def calculate_forward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int = 4,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        if num_links > KinematicsLaunchCfg.MAX_FORWARD_LINKS: raise RuntimeError("Forward kinematics kernel supports at most 500 links")
        max_threads, max_shared_mem = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS, max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        per = num_links * 96
        if per > max_shared_mem: raise RuntimeError("Single batch shared memory requirement exceeds limit")
        batches = max(1, min(batch_size, KinematicsLaunchCfg.MAX_FW_BATCH_PER_BLOCK, max_shared_mem // max(per, 1), max_threads // threads_per_batch))
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads_per_batch, batches * per)

    @staticmethod
    def calculate_backward_config(
        batch_size: int,
        num_links: int,
        num_spheres: int,
        n_tool_frames: int,
        n_joints: int,
        max_threads: int = None,
        max_shared_mem: int = None,
    ) -> Tuple[LaunchConfig, int, bool, int]:
        max_threads, max_shared_mem = max_threads or KinematicsLaunchCfg.DEFAULT_MAX_THREADS, max_shared_mem or KinematicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        per = num_links * 48
        if per > max_shared_mem: raise RuntimeError("Single batch shared memory requirement exceeds limit")
        use_warp = num_spheres < 5000
        max_joints = 16 if n_joints < 16 else 64 if n_joints < 64 else 128
        threads = 32 if use_warp else min(max_threads, max(32, num_spheres, n_tool_frames))
        batches = max(1, min(batch_size, KinematicsLaunchCfg.MAX_BW_BATCH_PER_BLOCK, max_threads // threads, max_shared_mem // max(per, 1))) if use_warp else 1
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads, batches * per), threads, use_warp, max_joints
