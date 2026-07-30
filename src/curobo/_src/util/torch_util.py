"""Portable torch feature and decorator helpers."""

from __future__ import annotations

import functools
import torch


def is_cuda_graph_available():
    return False


def is_cuda_graph_reset_available():
    return False


def is_torch_compile_available():
    return hasattr(torch, "compile")


def get_torch_compile_options():
    return {}


def disable_torch_compile_global():
    return None


def set_torch_compile_global_options():
    return None


def empty_decorator(function):
    return function


def get_torch_jit_decorator(
    force_jit: bool = False,
    dynamic: bool = True,
    only_valid_for_compile: bool = False,
    extreme_trace: bool = False,
    slow_to_compile: bool = False,
):
    del force_jit, dynamic, only_valid_for_compile, extreme_trace, slow_to_compile
    return empty_decorator


def get_profiler_decorator(str_name: str):
    return torch.autograd.profiler.record_function(str_name)


def profile_class_methods(cls):
    return cls
