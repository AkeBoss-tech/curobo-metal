"""Portable Torch compilation, JIT, and profiling helpers."""

from __future__ import annotations

import inspect
from functools import wraps

import torch
import torch.autograd.profiler as profiler
from packaging import version

from curobo import runtime as curobo_runtime
from curobo._src.util.logging import log_info, log_warn


def is_cuda_graph_available():
    if version.parse(torch.__version__) < version.parse("1.10"):
        log_warn("Disabling CUDA Graph as pytorch < 1.10")
        return False
    return True


def is_cuda_graph_reset_available():
    if not curobo_runtime.cuda_graph_reset:
        return False

    # The public CUDA-graph-reset capability is version-gated even though the
    # portable backends do not perform raw CUDA graph capture themselves.  A
    # CPU/MPS torch build reports ``None`` here; treat it as unavailable rather
    # than attempting to parse it as a packaging version.
    cuda_version = torch.version.cuda
    if cuda_version is not None and version.parse(cuda_version) >= version.parse("12.0"):
        return True
    log_warn("CUDA graph reset requires CUDA 12.0+, current version: " + str(cuda_version))
    return False


def is_torch_compile_available():
    if not curobo_runtime.torch_jit:
        return False
    if not curobo_runtime.torch_compile:
        log_info("torch.compile is explicitly disabled via runtime config")
        return False
    if version.parse(torch.__version__) < version.parse("2.0"):
        log_info("Disabling torch.compile as pytorch < 2.0")
        log_warn("Using pytorch < 2.0 is not recommended for performance reasons.")
        return False
    if not hasattr(torch, "compile"):
        log_info("Could not find torch.compile, disabling Torch Compile.")
        return False
    if not hasattr(torch, "_dynamo"):
        log_info("Could not find torch._dynamo, disabling Torch Compile.")
        return False
    log_info("torch.compile is available and enabled")
    return True


def get_torch_compile_options() -> dict:
    options = {}
    if is_torch_compile_available():
        from torch._inductor import config

        torch._dynamo.config.suppress_errors = True
        use_options = {
            "max_autotune": True,
            "use_mixed_mm": True,
            "conv_1x1_as_mm": True,
            "coordinate_descent_tuning": True,
            "epilogue_fusion": False,
            "coordinate_descent_check_all_directions": True,
            "force_fuse_int_mm_with_mul": True,
            "triton.cudagraphs": False,
            "aggressive_fusion": True,
            "split_reductions": False,
            "worker_start_method": "spawn",
        }
        for key, value in use_options.items():
            if hasattr(config, key):
                options[key] = value
            else:
                log_info("Not found in torch.compile: " + key)
    return options


def disable_torch_compile_global():
    if is_torch_compile_available():
        torch._dynamo.config.disable = True
        return True
    return False


def set_torch_compile_global_options():
    if not is_torch_compile_available():
        return False
    from torch._inductor import config

    torch._dynamo.config.suppress_errors = True
    for key, value in {
        "conv_1x1_as_mm": True,
        "coordinate_descent_tuning": True,
        "epilogue_fusion": False,
        "coordinate_descent_check_all_directions": True,
        "force_fuse_int_mm_with_mul": True,
        "use_mixed_mm": True,
    }.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return True


def empty_decorator(function):
    return function


def get_torch_jit_decorator(
    force_jit: bool = False,
    dynamic: bool = True,
    only_valid_for_compile: bool = False,
    extreme_trace: bool = False,
    slow_to_compile: bool = False,
):
    del extreme_trace, slow_to_compile
    if not curobo_runtime.torch_jit:
        return empty_decorator
    if not force_jit and is_torch_compile_available():
        return torch.compile(dynamic=dynamic)
    if not only_valid_for_compile:
        return torch.jit.script
    return empty_decorator


def get_profiler_decorator(str_name: str):
    if curobo_runtime.profiler:
        return profiler.record_function(str_name)
    return empty_decorator


def profile_class_methods(cls):
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("__"):
            continue

        @wraps(method)
        def wrapper(
            *args,
            _method=method,
            _label=f"{cls.__name__}/{name}",
            **kwargs,
        ):
            if not curobo_runtime.profiler:
                return _method(*args, **kwargs)
            with profiler.record_function(_label):
                return _method(*args, **kwargs)

        setattr(cls, name, wrapper)
    return cls
