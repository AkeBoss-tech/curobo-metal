from __future__ import annotations

import hashlib
from pathlib import Path


def get_cuda_home():
    return None


class CudaCoreKernelCache:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        raise NotImplementedError("CUDA kernel compilation is unavailable on Metal")

    def get_stream_wrapper(self, torch_stream):
        raise NotImplementedError("raw CUDA stream wrappers are unavailable on Metal")

    def get_kernel_hash(self, source_files, kernel_name, compile_flags):
        payload = kernel_name.encode()
        for path in source_files:
            payload += Path(path).read_bytes()
        payload += repr(tuple(compile_flags)).encode()
        return hashlib.sha256(payload).hexdigest()

    def get_or_compile_kernel(self, source_files, kernel_name, include_dirs, compile_flags):
        raise NotImplementedError("CUDA kernel compilation is unavailable on Metal")

    _compile_kernel = get_or_compile_kernel

    def _read_sources(self, source_files, include_dirs):
        return "\n".join(Path(path).read_text() for path in source_files)


cuda_debug_compile = False
