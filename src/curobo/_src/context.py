"""Portable global runtime cache matching pinned cuRoboV2."""


class CuroboRuntime:
    def __init__(self):
        self._cuda_core_cache = None
        self._warp_cache = None
        self._pybind_cache = None
        self.warp_runtime_kernel_cache = {}
        self.curobo_runtime_kernel_cache = {}

    def get_cuda_core_cache(self):
        raise NotImplementedError(
            "cuda.core kernel caching requires NVIDIA CUDA; Metal operators "
            "use their own shape-keyed runtime caches"
        )

    def get_warp_cache(self):
        raise NotImplementedError(
            "Warp kernel caching is unavailable in the portable Metal backend"
        )


runtime = None


def init():
    global runtime
    if runtime is None:
        runtime = CuroboRuntime()


def get_runtime() -> CuroboRuntime:
    if runtime is None:
        init()
    return runtime
