from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from curobo._src.util.logging import log_and_raise, log_debug, log_info, log_warn

try:
    from cuda import pathfinder as _cuda_pathfinder
except ImportError:  # Portable installs intentionally omit cuda-python.
    _cuda_pathfinder = None

# Keep the pinned module namespace importable without making cuda-python a
# Metal dependency. Raw CUDA compilation remains an explicit runtime boundary.
pathfinder = _cuda_pathfinder


def get_cuda_home() -> Optional[str]:
    if pathfinder is None:
        return None
    return pathfinder.find_nvidia_header_directory("nvrtc")


class CudaCoreKernelCache:
    def __init__(self):
        self.compiled_kernels: Dict[str, object] = {}
        self.device: Optional[object] = None
        self.arch: Optional[str] = None

    def initialize(self):
        raise NotImplementedError("CUDA kernel compilation is unavailable on Metal")

    def get_stream_wrapper(self, torch_stream):
        raise NotImplementedError("raw CUDA stream wrappers are unavailable on Metal")

    def get_kernel_hash(
        self, source_files: List[Path], kernel_name: str, compile_flags: List[str]
    ) -> str:
        content_hash = hashlib.sha256()
        for path in source_files:
            if path.exists():
                content_hash.update(path.read_bytes())
            else:
                log_warn(f"Source file not found: {path}")
        content_hash.update(kernel_name.encode())
        content_hash.update("_".join(compile_flags).encode())
        if self.arch:
            content_hash.update(self.arch.encode())
        return content_hash.hexdigest()

    def get_or_compile_kernel(
        self,
        source_files: List[Path],
        kernel_name: str,
        include_dirs: List[Path],
        compile_flags: List[str],
    ):
        raise NotImplementedError("CUDA kernel compilation is unavailable on Metal")

    _compile_kernel = get_or_compile_kernel

    def _read_sources(self, source_files: List[Path], include_dirs: List[Path]) -> str:
        return "\n".join(Path(path).read_text() for path in source_files)


cuda_debug_compile = False
