from pathlib import Path


class CudaCoreKernelCfg:
    def __init__(self, kernel_subdir: str):
        self._kernel_dir = Path(__file__).parent.parent.parent / "kernels" / kernel_subdir

    @property
    def kernel_dir(self):
        return self._kernel_dir

    def get_compile_flags(self, debug: bool = False):
        return ["-G", "-g"] if debug else ["-O3"]

    def get_base_include_dirs(self):
        return [self.kernel_dir.parent, self.kernel_dir.parent / "common", self.kernel_dir.parent / "third_party"]


cuda_debug_compile = False
