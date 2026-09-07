"""Construction-time memory guard for the dense-backed high-level Mapper."""

import pytest

from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg


def test_large_sparse_extent_fails_before_dense_allocation():
    """A sparse-sized map must not trigger a multi-gigabyte dense allocation."""
    config = MapperCfg(
        extent_meters_xyz=(10.0, 10.0, 10.0),
        voxel_size=0.02,
        block_size=8,
        device="cpu",
    )
    with pytest.raises(MemoryError, match="dense mirror.*safety limit"):
        Mapper(config)

