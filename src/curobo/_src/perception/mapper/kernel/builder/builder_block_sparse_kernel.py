from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias

from curobo._src.perception.mapper.constants import DEFAULT_HASH_LAYOUT, HashLayout
from curobo._src.perception.mapper.kernel.builder.builder_camera_integrate import make_camera_integrate_kernels
from curobo._src.perception.mapper.kernel.builder.builder_coord import make_coord_kernels
from curobo._src.perception.mapper.kernel.builder.builder_decay import make_decay_kernels
from curobo._src.perception.mapper.kernel.builder.builder_esdf import make_esdf_kernels
from curobo._src.perception.mapper.kernel.builder.builder_hash import make_hash_kernels
from curobo._src.perception.mapper.kernel.builder.builder_lidar_integrate import make_lidar_integrate_kernels
from curobo._src.perception.mapper.kernel.builder.builder_mesh import make_mesh_kernels
from curobo._src.perception.mapper.kernel.builder.builder_raycast import make_raycast_kernels
from curobo._src.perception.mapper.kernel.builder.builder_rescale import make_rescale_kernels
from curobo._src.perception.mapper.kernel.builder.builder_stamp import make_stamp_kernels
from curobo._src.util.logging import log_and_raise

annotations = None
WarpKernel: TypeAlias = Any
WarpFunction: TypeAlias = Any


@dataclass(frozen=True)
class BlockSparseKernels:
    """Marker describing the portable high-level backend.

    It deliberately contains no raw callable Warp kernels.
    """

    block_size: int
    backend: str = "torch"


def make_block_sparse_kernels(
    cfg: Any | None = None,
    *,
    block_size: int | None = None,
    seeding_method: str | None = None,
    feature_channels_per_thread: int | None = None,
) -> BlockSparseKernels:
    size = block_size if block_size is not None else getattr(cfg, "block_size", 8)
    return BlockSparseKernels(int(size))
