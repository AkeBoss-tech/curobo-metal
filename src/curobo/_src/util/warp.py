from __future__ import annotations

import hashlib
import torch

from curobo._src.types.device_cfg import DeviceCfg

wp = None
cuda_debug_compile = False


def _named(name, function):
    function.__name__ = function.__qualname__ = name
    return function


def warp_func(name):
    return lambda function: _named(name, function)


def warp_kernel(name, **kwargs):
    return lambda function: _named(name, function)


def warp_constant_suffix(*values):
    return hashlib.sha1(repr(values).encode("ascii")).hexdigest()[:12]


def init_warp(quiet=True, verbose=False, lineinfo=False, line_directives=False, print_launches=False, device_cfg=DeviceCfg()):
    raise NotImplementedError(
        "NVIDIA Warp is unavailable on Metal; use curobo-metal tensor operators"
    )


def _version_of(module):
    config = getattr(module, "config", None)
    return getattr(config, "version", "0.0.0")


def _at_least(module, minimum):
    from packaging.version import Version
    return Version(_version_of(module)) >= Version(minimum)


def warp_support_sdf_struct(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.0.0")
def warp_support_kernel_key(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.2.1")
def warp_support_bvh_constructor_type(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.6.0")


def get_warp_device_stream(tensor_or_device):
    device = tensor_or_device.device if isinstance(tensor_or_device, torch.Tensor) else torch.device(tensor_or_device)
    if device.type == "cuda":
        raise NotImplementedError("Warp CUDA stream interop is unavailable on Metal")
    return device, None
