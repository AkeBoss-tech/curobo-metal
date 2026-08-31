from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel
from curobo._src.perception.mapper.constants import DEFAULT_HASH_LAYOUT, HASH_EMPTY, HASH_PRIME_X, HASH_PRIME_Y, HASH_PRIME_Z, HASH_TOMBSTONE, HashLayout

ENTRY_EMPTY = HASH_EMPTY
ENTRY_TOMBSTONE = HASH_TOMBSTONE
annotations = None
warp_func = warp_kernel = wp = None


def make_hash_kernels(
    block_size: int,
    hash_layout: HashLayout = DEFAULT_HASH_LAYOUT,
) -> dict[str, object]:
    return unsupported_kernel(block_size, hash_layout)
