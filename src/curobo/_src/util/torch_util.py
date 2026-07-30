"""Portable Torch compilation, JIT, and profiling helpers."""

from __future__ import annotations

import inspect
from functools import wraps

import torch
import torch.autograd.profiler as profiler

from curobo import runtime as curobo_runtime


def is_cuda_graph_available():
    return False


def is_cuda_graph_reset_available():
    return False


def is_torch_compile_available():
    return bool(curobo_runtime.torch_compile and hasattr(torch, "compile"))


def get_torch_compile_options():
    return {}


def disable_torch_compile_global():
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo.config, "disable"):
        torch._dynamo.config.disable = True
        return True
    return False


def set_torch_compile_global_options():
    return is_torch_compile_available()


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
