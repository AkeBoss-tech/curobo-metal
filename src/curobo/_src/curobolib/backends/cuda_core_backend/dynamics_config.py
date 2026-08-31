from pathlib import Path
from typing import List

from ._launch import LaunchConfig
from .kernel_config import CudaCoreKernelCfg
from .util import ceil_div


class DynamicsKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("dynamics")
    def get_kernel_files(self, kernel_type: str) -> List[str]:
        return {"forward": ["rnea_forward_kernel.cuh"], "backward": ["rnea_backward_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self) -> List[Path]: return self.get_base_include_dirs() + [self.kernel_dir, self.kernel_dir.parent / "kinematics"]


class DynamicsLaunchCfg:
    DEFAULT_MAX_BATCHES_PER_BLOCK = 256
    DEFAULT_MAX_BW_BATCHES_PER_BLOCK = 256
    DEFAULT_MAX_SHARED_MEM = 48 * 1024
    DEFAULT_SM_SHARED_MEM_CAPACITY = 100 * 1024
    WARP_SIZE = 32

    @staticmethod
    def _warp_align_batches(
        batches_per_block: int,
        threads_per_batch: int,
        smem_per_block_fn,
        sm_shared_mem_capacity: int,
    ) -> int:
        granularity = max(1, DynamicsLaunchCfg.WARP_SIZE // threads_per_batch)
        candidate = batches_per_block // granularity * granularity
        if candidate < granularity:
            return batches_per_block
        candidate_lower = candidate - granularity
        if candidate_lower < granularity:
            candidate_lower = 0
        best_batches, best_score = candidate, 0
        for cand in (candidate, candidate_lower):
            if cand < 1:
                continue
            smem = smem_per_block_fn(cand)
            blocks_per_sm = sm_shared_mem_capacity // smem if smem > 0 else 1
            score = max(1, blocks_per_sm) * cand * threads_per_batch
            if score > best_score:
                best_batches, best_score = cand, score
        return best_batches

    @staticmethod
    def calculate_forward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int = 1,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        max_bpb = max_batches_per_block or DynamicsLaunchCfg.DEFAULT_MAX_BATCHES_PER_BLOCK
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        smem_per_batch = num_links * 12 * 4
        if smem_per_batch > max_shared_mem:
            raise RuntimeError("Single batch shared memory requirement exceeds limit")
        batches = max(1, min(max_shared_mem // smem_per_batch, 1024 // threads_per_batch, max_bpb, batch_size))
        batches = DynamicsLaunchCfg._warp_align_batches(
            batches,
            threads_per_batch,
            lambda b: b * smem_per_batch,
            DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY,
        )
        return LaunchConfig(ceil_div(batch_size, batches), batches * threads_per_batch, batches * smem_per_batch)

    @staticmethod
    def calculate_backward_config(
        batch_size: int,
        num_links: int,
        threads_per_batch: int = 1,
        max_batches_per_block: int = None,
        max_shared_mem: int = None,
    ) -> LaunchConfig:
        max_bpb = max_batches_per_block or DynamicsLaunchCfg.DEFAULT_MAX_BW_BATCHES_PER_BLOCK
        max_shared_mem = max_shared_mem or DynamicsLaunchCfg.DEFAULT_MAX_SHARED_MEM
        smem_block_shared = num_links * 12 * 4
        smem_per_batch = num_links * 30 * 4
        if smem_block_shared + smem_per_batch > max_shared_mem:
            raise RuntimeError("Single batch shared memory requirement exceeds limit")
        batches = max(1, min(
            (max_shared_mem - smem_block_shared) // smem_per_batch,
            1024 // threads_per_batch,
            max_bpb,
            batch_size,
        ))
        batches = DynamicsLaunchCfg._warp_align_batches(
            batches,
            threads_per_batch,
            lambda b: smem_block_shared + b * smem_per_batch,
            DynamicsLaunchCfg.DEFAULT_SM_SHARED_MEM_CAPACITY,
        )
        return LaunchConfig(
            ceil_div(batch_size, batches),
            batches * threads_per_batch,
            smem_block_shared + batches * smem_per_batch,
        )
