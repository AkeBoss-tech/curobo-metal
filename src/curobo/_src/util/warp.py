from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Optional, Tuple, Union

import torch
from packaging import version

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_debug, log_info

wp = None
_warp_module = None
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
    """Initialize the optional Warp runtime when a usable build is present.

    Warp ships a CPU backend that is useful for the upstream compatibility
    tests and for host-side helpers.  Metal tensors still cannot be passed to
    Warp kernels, but that limitation belongs at the stream/kernel boundary,
    not at process initialization.  In particular, CUDA-oriented test
    fixtures call this helper before deciding whether CUDA is available; an
    unavailable CUDA device must therefore not turn a later, intentional
    pytest skip into a collection error.
    """
    del quiet, verbose, lineinfo, line_directives, print_launches, device_cfg
    global _warp_module, _portable_initialized
    if _portable_initialized and _warp_module is not None:
        return True
    try:
        import warp as warp_module
    except ImportError as error:
        raise ImportError("Warp is not installed") from error
    # warp.init() is idempotent and uses the devices available in this build.
    # Do not request the caller's CUDA/MPS device here: a CPU-only Warp wheel
    # is still a valid initialization for high-level portable code.
    warp_module.init()
    # Keep ``wp`` as the portable declaration sentinel.  Several compatibility
    # modules intentionally inspect it at import time and install lightweight
    # decorators instead of asking Warp to compile CUDA-only struct types.
    _warp_module = warp_module
    _portable_initialized = True
    log_debug(f"Warp initialized - Version: {warp_module.config.version}")
    log_info("Warp initialized for portable host-side compatibility")
    return True


def _version_of(module):
    config = getattr(module, "config", None)
    return getattr(config, "version", "0.0.0")


def _at_least(module, minimum):
    return version.parse(_version_of(module)) >= version.parse(minimum)


def warp_support_sdf_struct(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.0.0")
def warp_support_kernel_key(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.2.1")
def warp_support_bvh_constructor_type(wp_module=None): return False if wp_module is None else _at_least(wp_module, "1.6.0")


def get_warp_device_stream(
    tensor_or_device: Union[torch.Tensor, torch.device],
) -> Tuple[wp.Device, Optional[wp.Stream]]:
    """Return a real Warp device for CPU tensors without hiding MPS gaps."""
    device = tensor_or_device.device if isinstance(tensor_or_device, torch.Tensor) else torch.device(tensor_or_device)
    if device.type == "mps":
        raise NotImplementedError("Warp stream interop is unavailable for MPS tensors")
    if _warp_module is None:
        init_warp(device_cfg=DeviceCfg(device=device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise NotImplementedError("Warp CUDA stream interop is unavailable on this build")
    try:
        # Warp's Python API accepts a string identifier (``"cpu"`` or
        # ``"cuda:0"``), not a torch.device instance.
        return _warp_module.get_device(str(device)), None
    except Exception as error:
        raise NotImplementedError(
            f"Warp does not provide a usable {device} device"
        ) from error
