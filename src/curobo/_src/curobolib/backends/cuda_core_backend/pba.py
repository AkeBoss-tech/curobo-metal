import torch

from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel

from .pba_config import PBAKernelCfg, PBALaunchCfg


def launch_pba3d(
    site_index: torch.Tensor,
    buffer: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    m3: int = 2,
) -> None:
    raise NotImplementedError(
        "PBA+ is a raw CUDA EDT kernel; use curobo.perception.Mapper.compute_esdf "
        "for the portable exact dense ESDF"
    )
