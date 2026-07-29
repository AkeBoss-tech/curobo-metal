"""Portable robotics primitives for cuRobo Metal.

Backend symbols are loaded lazily so configuration-only consumers can use the
compatibility layer without importing PyTorch or initializing a GPU runtime.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Backend", "resolve_device", "synchronize", "validate_tensor_device"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import backend

    return getattr(backend, name)
