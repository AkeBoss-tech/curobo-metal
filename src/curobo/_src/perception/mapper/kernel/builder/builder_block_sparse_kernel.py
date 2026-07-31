from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSparseKernels:
    """Marker describing the portable high-level backend.

    It deliberately contains no raw callable Warp kernels.
    """

    block_size: int
    backend: str = "torch"


def make_block_sparse_kernels(cfg=None, *, block_size=None, seeding_method=None, feature_channels_per_thread=None):
    size = block_size if block_size is not None else getattr(cfg, "block_size", 8)
    return BlockSparseKernels(int(size))
