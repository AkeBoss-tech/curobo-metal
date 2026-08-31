"""Pinned marching-cubes declarations with an explicit Warp ABI boundary.

The upstream module exposes lookup data and Warp device functions.  The
high-level mapper has a portable CPU/MPS implementation, while this raw
CUDA/Warp-facing surface must remain importable and must reject execution
deterministically when Warp is unavailable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.util.warp import init_warp, wp

try:  # Do not install or register a substitute ``warp`` module globally.
    import warp as _raw_wp
except ImportError:
    _raw_wp = None


if _raw_wp is not None:
    wp = _raw_wp
elif wp is None:

    class _PortableWarp:
        """Declaration-only subset used to retain the upstream Python surface."""

        int32 = int
        int64 = int
        float32 = float
        vec3 = tuple

        @staticmethod
        def array(*args, **kwargs):
            return object

        @staticmethod
        def func(function):
            return function

    wp = _PortableWarp()


# These raw lookup tables are consumed only by the Warp implementation.  The
# portable mesh extractor keeps its own CPU/MPS representation; keeping the
# public bindings here makes the raw boundary inspectable without claiming a
# Warp array ABI on Metal.
TRIANGLE_TABLE = ()
NUM_TRIANGLES_TABLE = ()
EDGE_OWNER_OFFSETS = ()


@wp.func
def local_edge_to_array_idx(local_edge: wp.int32) -> wp.int32:
    return unsupported_kernel(local_edge)


@wp.func
def array_idx_to_local_edge(array_idx: wp.int32) -> wp.int32:
    return unsupported_kernel(array_idx)


@wp.func
def binary_search_int32(
    sorted_keys: wp.array(dtype=wp.int32),
    n: wp.int32,
    target: wp.int32,
) -> wp.int32:
    return unsupported_kernel(sorted_keys, n, target)


@wp.func
def binary_search_int64(
    sorted_keys: wp.array(dtype=wp.int64),
    n: wp.int32,
    target: wp.int64,
) -> wp.int32:
    return unsupported_kernel(sorted_keys, n, target)


@wp.func
def interpolate_edge_vertex(
    p0: wp.vec3,
    p1: wp.vec3,
    s0: wp.float32,
    s1: wp.float32,
) -> wp.vec3:
    return unsupported_kernel(p0, p1, s0, s1)


@wp.func
def get_edge_vertex(
    edge_id: wp.int32,
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    p4: wp.vec3,
    p5: wp.vec3,
    p6: wp.vec3,
    p7: wp.vec3,
    s0: wp.float32,
    s1: wp.float32,
    s2: wp.float32,
    s3: wp.float32,
    s4: wp.float32,
    s5: wp.float32,
    s6: wp.float32,
    s7: wp.float32,
) -> wp.vec3:
    return unsupported_kernel(
        edge_id,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
    )


class MCLookupTables:
    """Per-device raw Warp marching-cubes lookup-table cache."""

    _instances: Dict[str, "MCLookupTables"] = {}

    def __init__(self, device: torch.device):
        # ``init_warp`` consistently raises on the portable runtime.  Retain
        # that rejection rather than producing misleading host-side raw arrays.
        init_warp()
        self.device = device

    @classmethod
    def get(cls, device: torch.device) -> "MCLookupTables":
        key = str(device)
        if key not in cls._instances:
            cls._instances[key] = cls(device)
        return cls._instances[key]
