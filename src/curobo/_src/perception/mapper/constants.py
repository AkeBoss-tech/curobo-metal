"""Shared host-side constants for the portable block-sparse mapper.

The public declarations deliberately mirror cuRobo's Warp-facing module. On
CPU/MPS the scalar ``wp.constant`` values remain ordinary Python scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import warp as _wp
except ImportError:
    class _PortableWarp:
        int32 = int
        int64 = int

        @staticmethod
        def constant(value):
            return value

    _wp = _PortableWarp()

wp = _wp

from curobo._src.util.logging import log_and_raise


REFERENCE_BLOCK_SIZE: int = 8
_BLOCK_SIZE_MAX: int = 32
FEATURE_INTEGRATION_KERNEL_MODES: tuple[str, str, str] = ("auto", "grouped", "tiled")


def _validate_block_size(block_size: int) -> None:
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        log_and_raise(f"block_size must be a plain int, got {type(block_size).__name__} ({block_size!r}).")
    if block_size < 1:
        log_and_raise(f"block_size must be >= 1, got {block_size}.")
    if block_size & (block_size - 1) != 0:
        log_and_raise(f"block_size must be 1 or a power of 2 (2, 4, 8, 16, 32), got {block_size}.")
    if block_size > _BLOCK_SIZE_MAX:
        log_and_raise(f"block_size must be <= {_BLOCK_SIZE_MAX}, got {block_size}.")


def _validate_block_grid_size(grid_size: int, block_size: int, field_name: str) -> None:
    if not isinstance(grid_size, int) or isinstance(grid_size, bool):
        log_and_raise(f"{field_name} must be a plain int, got {type(grid_size).__name__} ({grid_size!r}).")
    if grid_size < 1:
        log_and_raise(f"{field_name} must be >= 1, got {grid_size}.")
    if grid_size > block_size:
        log_and_raise(f"{field_name} must be <= block_size, got {field_name}={grid_size}, block_size={block_size}.")


def _validate_color_grid_size(color_grid_size: int, block_size: int) -> None:
    _validate_block_grid_size(color_grid_size, block_size, "color_grid_size")


def _validate_feature_block_grid_size(feature_block_grid_size: int, block_size: int) -> None:
    _validate_block_grid_size(feature_block_grid_size, block_size, "feature_block_grid_size")


def _validate_feature_channels_per_thread(feature_channels_per_thread: int) -> None:
    if not isinstance(feature_channels_per_thread, int) or isinstance(feature_channels_per_thread, bool):
        log_and_raise("feature_channels_per_thread must be a plain int.")
    if feature_channels_per_thread < 1:
        log_and_raise("feature_channels_per_thread must be >= 1.")


def _validate_feature_grid_shape(
    feature_dim: int, feature_grid_height: int | None, feature_grid_width: int | None
) -> None:
    if feature_dim < 0:
        log_and_raise(f"feature_dim must be >= 0, got {feature_dim}.")
    has_height, has_width = feature_grid_height is not None, feature_grid_width is not None
    if has_height != has_width:
        log_and_raise("feature_grid_height and feature_grid_width must be specified together.")
    if feature_dim == 0:
        if has_height:
            log_and_raise("feature_grid_height/feature_grid_width require feature_dim > 0.")
        return
    if not has_height:
        log_and_raise("feature_dim > 0 requires feature_grid_height and feature_grid_width.")
    if not isinstance(feature_grid_height, int) or isinstance(feature_grid_height, bool):
        log_and_raise("feature_grid_height must be a plain int.")
    if not isinstance(feature_grid_width, int) or isinstance(feature_grid_width, bool):
        log_and_raise("feature_grid_width must be a plain int.")
    if feature_grid_height <= 0 or feature_grid_width <= 0:
        log_and_raise("feature_grid_height and feature_grid_width must be positive.")


def _validate_lidar_config(
    lidar_num_sensors: int,
    lidar_image_height: int | None,
    lidar_image_width: int | None,
    lidar_feature_grid_height: int | None,
    lidar_feature_grid_width: int | None,
    feature_dim: int,
    lidar_linear_interpolation_max_allowable_difference_vox: float,
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox: float,
    max_support_pixels_per_block_lidar: int,
) -> None:
    if lidar_num_sensors < 0:
        log_and_raise("lidar_num_sensors must be >= 0.")
    if lidar_num_sensors:
        raise NotImplementedError("lidar mapping is unavailable in the portable mapper")


def _validate_feature_integration_kernel(feature_integration_kernel: str) -> None:
    if not isinstance(feature_integration_kernel, str) or feature_integration_kernel not in FEATURE_INTEGRATION_KERNEL_MODES:
        log_and_raise(f"feature_integration_kernel must be one of {FEATURE_INTEGRATION_KERNEL_MODES}, got {feature_integration_kernel!r}.")


def resolve_feature_integration_kernel(
    feature_integration_kernel: str,
    feature_dim: int,
    support_capacity: int,
) -> bool:
    _validate_feature_integration_kernel(feature_integration_kernel)
    if feature_integration_kernel == "tiled":
        return True
    if feature_integration_kernel == "grouped":
        return False
    return (
        feature_dim >= 512
        or (feature_dim >= 128 and support_capacity <= 8)
        or (feature_dim >= 64 and support_capacity <= 4)
    )


@dataclass(frozen=True)
class HashLayout:
    coord_bits_xyz: Tuple[int, int, int]

    def __post_init__(self) -> None:
        bits = self.coord_bits_xyz
        if len(bits) != 3 or any(not isinstance(bit_count, int) or isinstance(bit_count, bool) or bit_count <= 0 for bit_count in bits):
            raise ValueError(f"coord_bits_xyz must contain three positive integers, got {bits!r}.")
        if sum(bits) >= 64:
            raise ValueError(f"HashLayout must leave at least one value bit; got coord_bits_xyz={bits!r}.")
        if self.pending_pool_idx > 2_147_483_647:
            raise ValueError("HashLayout pending_pool_idx must fit int32 for Warp kernels.")

    @property
    def value_bits(self) -> int:
        return 64 - sum(self.coord_bits_xyz)

    @property
    def name(self) -> str:
        x_bits, y_bits, z_bits = self.coord_bits_xyz
        return f"x{x_bits}y{y_bits}z{z_bits}v{self.value_bits}"

    @property
    def coord_masks_xyz(self) -> Tuple[int, int, int]:
        return tuple((1 << bits) - 1 for bits in self.coord_bits_xyz)

    @property
    def coord_bias_xyz(self) -> Tuple[int, int, int]:
        return tuple(1 << (bits - 1) for bits in self.coord_bits_xyz)

    @property
    def coord_min_xyz(self) -> Tuple[int, int, int]:
        return tuple(-bias for bias in self.coord_bias_xyz)

    @property
    def coord_max_xyz(self) -> Tuple[int, int, int]:
        return tuple(bias - 1 for bias in self.coord_bias_xyz)

    @property
    def value_mask(self) -> int:
        return (1 << self.value_bits) - 1

    @property
    def key_mask(self) -> int:
        return ((1 << 64) - 1) ^ self.value_mask

    @property
    def key_mask_signed(self) -> int:
        return -1 << self.value_bits

    @property
    def z_shift(self) -> int:
        return self.value_bits

    @property
    def y_shift(self) -> int:
        return self.z_shift + self.coord_bits_xyz[2]

    @property
    def x_shift(self) -> int:
        return self.y_shift + self.coord_bits_xyz[1]

    @property
    def pending_pool_idx(self) -> int:
        return self.value_mask

    @property
    def max_pool_idx(self) -> int:
        return self.pending_pool_idx - 1


DEFAULT_HASH_LAYOUT = HashLayout(coord_bits_xyz=(13, 13, 13))


def _ceil_div_positive(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def validate_grid_shape_for_hash_layout(
    grid_shape: Tuple[int, int, int] | None,
    block_size: int,
    *,
    layout: HashLayout = DEFAULT_HASH_LAYOUT,
    field_name: str = "grid_shape",
) -> Tuple[int, int, int]:
    if grid_shape is None:
        log_and_raise(f"{field_name} is required for bounded block-sparse mapping.")
    if len(grid_shape) != 3:
        log_and_raise(f"{field_name} must be a 3-tuple (nz, ny, nx), got {grid_shape!r}.")
    try:
        nz, ny, nx = (int(v) for v in grid_shape)
    except (TypeError, ValueError):
        log_and_raise(f"{field_name} must contain integer dimensions, got {grid_shape!r}.")
    if nz <= 0 or ny <= 0 or nx <= 0:
        log_and_raise(f"{field_name} dimensions must be positive, got {(nz, ny, nx)}.")
    block_counts = (_ceil_div_positive(nx, block_size), _ceil_div_positive(ny, block_size), _ceil_div_positive(nz, block_size))
    for axis, count, limit in zip(("x", "y", "z"), block_counts, (1 << bits for bits in layout.coord_bits_xyz)):
        if count > limit:
            log_and_raise(f"{field_name} requires {count} blocks along {axis}, exceeding the {layout.name} hash layout limit of {limit} blocks/axis.")
    return (nz, ny, nx)


PY_HASH_EMPTY: int = -1
PY_HASH_TOMBSTONE: int = -2
HASH_EMPTY = wp.constant(wp.int64(PY_HASH_EMPTY))
HASH_TOMBSTONE = wp.constant(wp.int64(PY_HASH_TOMBSTONE))
PY_HASH_PRIME_X: int = 73856093
PY_HASH_PRIME_Y: int = 19349663
PY_HASH_PRIME_Z: int = 83492791
HASH_PRIME_X = wp.constant(wp.int64(PY_HASH_PRIME_X))
HASH_PRIME_Y = wp.constant(wp.int64(PY_HASH_PRIME_Y))
HASH_PRIME_Z = wp.constant(wp.int64(PY_HASH_PRIME_Z))
PY_COORD_BITS: int = DEFAULT_HASH_LAYOUT.coord_bits_xyz[0]
PY_COORD_OFFSET: int = DEFAULT_HASH_LAYOUT.coord_bias_xyz[0]
PY_COORD_MASK: int = DEFAULT_HASH_LAYOUT.coord_masks_xyz[0]
COORD_BITS = wp.constant(PY_COORD_BITS)
COORD_OFFSET = wp.constant(wp.int64(PY_COORD_OFFSET))
COORD_MASK = wp.constant(wp.int64(PY_COORD_MASK))
BLOCK_KEY_BITS = COORD_BITS
BLOCK_KEY_OFFSET = COORD_OFFSET
BLOCK_KEY_MASK = COORD_MASK
PY_VALUE_BITS: int = DEFAULT_HASH_LAYOUT.value_bits
PY_VALUE_MASK: int = DEFAULT_HASH_LAYOUT.value_mask
VALUE_BITS = wp.constant(PY_VALUE_BITS)
VALUE_MASK = wp.constant(wp.int64(PY_VALUE_MASK))
PY_Z_SHIFT: int = DEFAULT_HASH_LAYOUT.z_shift
PY_Y_SHIFT: int = DEFAULT_HASH_LAYOUT.y_shift
PY_X_SHIFT: int = DEFAULT_HASH_LAYOUT.x_shift
Z_SHIFT = wp.constant(wp.int64(PY_Z_SHIFT))
Y_SHIFT = wp.constant(wp.int64(PY_Y_SHIFT))
X_SHIFT = wp.constant(wp.int64(PY_X_SHIFT))
PY_KEY_MASK: int = DEFAULT_HASH_LAYOUT.key_mask
KEY_MASK = wp.constant(wp.int64(DEFAULT_HASH_LAYOUT.key_mask_signed))
PY_POSITIVE_MASK: int = 0x7FFFFFFFFFFFFFFF
PENDING_POOL_IDX: int = DEFAULT_HASH_LAYOUT.pending_pool_idx
PENDING_POOL_IDX_WP = wp.constant(wp.int32(PENDING_POOL_IDX))
# The upstream packed key admits a much larger CUDA pool.  The portable
# storage implementation deliberately retains its established bounded pool
# ceiling, preventing allocations that cannot be materialized on MPS.
MAX_POOL_IDX: int = min(DEFAULT_HASH_LAYOUT.max_pool_idx, 65_534)
