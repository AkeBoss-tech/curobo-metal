from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class DynamicsKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("dynamics")
    def get_kernel_files(self, kernel_type: str):
        return {"forward": ["rnea_forward_kernel.cuh"], "backward": ["rnea_backward_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self): return self.get_base_include_dirs() + [self.kernel_dir, self.kernel_dir.parent / "kinematics"]


class DynamicsLaunchCfg:
    @staticmethod
    def _warp_align_batches(batches_per_block, threads_per_batch, smem_per_block_fn, sm_shared_mem_capacity):
        granularity = max(1, 32 // threads_per_batch)
        return batches_per_block if batches_per_block < granularity else max(granularity, batches_per_block // granularity * granularity)

    @staticmethod
    def _config(batch_size, num_links, threads_per_batch, backward, max_batches_per_block, max_shared_mem):
        max_shared_mem = max_shared_mem or 48 * 1024
        shared = num_links * (42 if backward else 12) * 4
        if shared > max_shared_mem: raise RuntimeError("Single batch shared memory requirement exceeds limit")
        batches = max(1, min(batch_size, max_batches_per_block or 256, 1024 // threads_per_batch, max_shared_mem // max(shared, 1)))
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads_per_batch, batches * shared)

    @staticmethod
    def calculate_forward_config(batch_size, num_links, threads_per_batch=1, max_batches_per_block=None, max_shared_mem=None):
        return DynamicsLaunchCfg._config(batch_size, num_links, threads_per_batch, False, max_batches_per_block, max_shared_mem)

    @staticmethod
    def calculate_backward_config(batch_size, num_links, threads_per_batch=1, max_batches_per_block=None, max_shared_mem=None):
        return DynamicsLaunchCfg._config(batch_size, num_links, threads_per_batch, True, max_batches_per_block, max_shared_mem)
