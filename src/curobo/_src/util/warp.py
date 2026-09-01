from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Optional, Tuple, Union

import torch
from packaging import version

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_debug, log_info

wp = None
cuda_debug_compile = False
_portable_initialized = False


def _named(name, function):
    function.__name__ = function.__qualname__ = name
    return function


def warp_func(name: str) -> Callable[[Callable[..., Any]], Any]:
    return lambda function: _named(name, function)


def warp_kernel(name: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    return lambda function: _named(name, function)


def warp_constant_suffix(*values: object) -> str:
    return hashlib.sha1(repr(values).encode("ascii")).hexdigest()[:12]


def init_warp(
    quiet=True,
    verbose=False,
    lineinfo=False,
    line_directives=False,
    print_launches=False,
    device_cfg: DeviceCfg = DeviceCfg()
):
    """Initialize the portable compatibility layer once.

    High-level modules call this as a process bootstrap even when their Metal
    implementation uses only Torch tensors.  Successful initialization does
    not claim that raw Warp kernels or CUDA stream interop are available.
    """
    del quiet, verbose, lineinfo, line_directives, print_launches, device_cfg
    global _portable_initialized
    _portable_initialized = True
    return True


def _version_of(module):
    config = getattr(module, "config", None)
    return getattr(config, "version", "0.0.0")


def _at_least(module, minimum):
    from packaging.version import Version
    return Version(_version_of(module)) >= Version(minimum)


def warp_support_sdf_struct(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.0.0")
def warp_support_kernel_key(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.2.1")
def warp_support_bvh_constructor_type(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.6.0")


def get_warp_device_stream(
    tensor_or_device: Union[torch.Tensor, torch.device],
) -> Tuple[wp.Device, Optional[wp.Stream]]:
    device = tensor_or_device.device if isinstance(tensor_or_device, torch.Tensor) else torch.device(tensor_or_device)
    if device.type == "cuda":
        raise NotImplementedError("Warp CUDA stream interop is unavailable on Metal")
    return device, None
