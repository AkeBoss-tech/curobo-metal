from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class KinematicsKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("kinematics")
    def get_kernel_files(self, kernel_type: str):
        return {"forward": ["kinematics_forward_kernel.cuh"], "backward": ["kinematics_backward_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self): return self.get_base_include_dirs() + [self.kernel_dir]


class KinematicsLaunchCfg:
    @staticmethod
    def calculate_forward_config(batch_size, num_links, threads_per_batch=4, max_threads=None, max_shared_mem=None):
        if num_links > 500: raise RuntimeError("Forward kinematics kernel supports at most 500 links")
        max_threads, max_shared_mem = max_threads or 128, max_shared_mem or 48 * 1024
        per = num_links * 96
        if per > max_shared_mem: raise RuntimeError("Single batch shared memory requirement exceeds limit")
        batches = max(1, min(batch_size, 8, max_shared_mem // max(per, 1), max_threads // threads_per_batch))
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads_per_batch, batches * per)

    @staticmethod
    def calculate_backward_config(batch_size, num_links, num_spheres, n_tool_frames, n_joints, max_threads=None, max_shared_mem=None):
        max_threads, max_shared_mem = max_threads or 128, max_shared_mem or 48 * 1024
        per = num_links * 48
        if per > max_shared_mem: raise RuntimeError("Single batch shared memory requirement exceeds limit")
        use_warp = num_spheres < 5000
        max_joints = 16 if n_joints < 16 else 64 if n_joints < 64 else 128
        threads = 32 if use_warp else min(max_threads, max(32, num_spheres, n_tool_frames))
        batches = max(1, min(batch_size, max_threads // threads, max_shared_mem // max(per, 1))) if use_warp else 1
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads, batches * per), threads, use_warp, max_joints
