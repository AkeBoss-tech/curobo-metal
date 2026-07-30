"""Backend selection without importing CUDA-only packages."""

from __future__ import annotations

from importlib import import_module


class _BackendProxy:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, attr: str):
        module = import_module(
            f"curobo._src.curobolib.backends.cuda_core_backend.{self.name}"
        )
        return getattr(module, attr)


def get_backend_name():
    return "portable"


def get_backend():
    return {name: _BackendProxy(name) for name in _BackendProxy._MODULES}


enable_cuda_core_runtime = False

def __getattr__(name):
    if name in _BackendProxy._MODULES:
        return _BackendProxy(name)
    raise AttributeError(name)


_BackendProxy._MODULES = frozenset(
    {"kinematics", "optimization", "trajectory", "geometry", "dynamics", "pba"}
)

__all__ = ["get_backend", "get_backend_name", "enable_cuda_core_runtime"]
