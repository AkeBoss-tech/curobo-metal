"""Mutable cuRobo runtime flags.

CUDA-named flags are retained for configuration compatibility. Portable
solvers compile them into Metal/PyTorch cache behavior or reject unsupported
low-level capture requests explicitly.
"""

from pathlib import Path

torch_compile = False
torch_compile_slow = False
torch_jit = False
cuda_graphs = True
cuda_graph_reset = False
cuda_streams = True
cuda_event_timers = True
cuda_core_backend = True
kernel_backend = "auto"
cache_dir = str(Path.home() / ".cache" / "curobo")
debug = False
debug_cuda_graphs = False
debug_cuda_compile = False
debug_nan = False
debug_timers = False
debug_trajopt = False
profiler = False
