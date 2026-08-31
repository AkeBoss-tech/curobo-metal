from pathlib import Path
from typing import List

from curobo._src.runtime import debug_cuda_compile as cuda_debug_compile


class CudaCoreKernelCfg:
    def __init__(self, kernel_subdir: str):
        self._kernel_dir = Path(__file__).parent.parent.parent / "kernels" / kernel_subdir

    @property
    def kernel_dir(self) -> Path:
        return self._kernel_dir

    def get_compile_flags(self, debug: bool = False) -> List[str]:
        if debug or cuda_debug_compile:
            return ["-G", "-g", "--generate-line-info", "--device-debug"]
        return [
            "-O3",
            "--ftz=true",
            "--fmad=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "--generate-line-info",
        ]

    def get_base_include_dirs(self) -> List[Path]:
        return [self.kernel_dir.parent, self.kernel_dir.parent / "common", self.kernel_dir.parent / "third_party"]
