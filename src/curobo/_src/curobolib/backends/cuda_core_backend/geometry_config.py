from pathlib import Path
from typing import List

from .kernel_config import CudaCoreKernelCfg


class GeometryKernelCfg(CudaCoreKernelCfg):
    def __init__(self): super().__init__("geometry")
    def get_kernel_files(self, kernel_type: str) -> List[str]:
        return {"self_collision": ["self_collision/self_collision_kernel.cuh"]}.get(kernel_type, [])
    def get_include_dirs(self) -> List[Path]:
        return self.get_base_include_dirs() + [self.kernel_dir, self.kernel_dir / "common", self.kernel_dir / "self_collision"]
