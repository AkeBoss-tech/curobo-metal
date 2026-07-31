"""Portable exact dense ESDF integration."""

from dataclasses import dataclass

from curobo_metal.ops.perception import dense_esdf

from ._portable import dense_state


@dataclass
class BlockSparseESDFIntegratorCfg:
    voxel_size: float = 0.05
    grid_shape: tuple[int, int, int] | None = None
    empty_value: float = 1.0
    minimum_tsdf_weight: float = 0.1
    device: str = "cuda:0"


class BlockSparseESDFIntegrator:
    def __init__(self, cfg: BlockSparseESDFIntegratorCfg | None = None, **kwargs):
        self.cfg = cfg or BlockSparseESDFIntegratorCfg(**kwargs)

    def compute(self, tsdf):
        state = dense_state(tsdf)
        return dense_esdf(
            state.occupancy, self.cfg.voxel_size, self.cfg.empty_value,
            dtype=state.tsdf.dtype,
        )

    integrate = compute
    __call__ = compute
