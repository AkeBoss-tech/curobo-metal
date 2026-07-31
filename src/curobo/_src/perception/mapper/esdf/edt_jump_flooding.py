"""Exact portable replacement for the approximate CUDA jump-flooding EDT."""

from curobo_metal.ops.perception import dense_esdf


class JumpFloodingEDT:
    def __init__(self, grid_shape=None, voxel_size=1.0, empty_value=1.0, **kwargs):
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.empty_value = empty_value

    def compute(self, occupancy, *args, **kwargs):
        return dense_esdf(
            occupancy.bool(), self.voxel_size, self.empty_value,
            dtype=kwargs.get("dtype", occupancy.new_empty(()).float().dtype),
        )

    __call__ = compute
