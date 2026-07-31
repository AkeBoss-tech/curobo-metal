"""Portable constants and validation for the pinned block-map layout."""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_BLOCK_SIZE = 8
FEATURE_INTEGRATION_KERNEL_MODES = ("auto", "grouped", "tiled")


@dataclass(frozen=True)
class HashLayout:
    name: str = "portable-int64"
    coord_bits: int = 16
    value_bits: int = 16
    max_pool_idx: int = 65534


DEFAULT_HASH_LAYOUT = HashLayout()
PY_HASH_EMPTY = HASH_EMPTY = -1
PY_HASH_TOMBSTONE = HASH_TOMBSTONE = -2
PY_HASH_PRIME_X = HASH_PRIME_X = 73856093
PY_HASH_PRIME_Y = HASH_PRIME_Y = 19349663
PY_HASH_PRIME_Z = HASH_PRIME_Z = 83492791
PY_COORD_BITS = COORD_BITS = 16
PY_COORD_OFFSET = COORD_OFFSET = 1 << 15
PY_COORD_MASK = COORD_MASK = (1 << 16) - 1
BLOCK_KEY_BITS = 48
BLOCK_KEY_OFFSET = 0
BLOCK_KEY_MASK = (1 << 48) - 1
PY_VALUE_BITS = VALUE_BITS = 16
PY_VALUE_MASK = VALUE_MASK = (1 << 16) - 1
PY_Z_SHIFT = Z_SHIFT = 0
PY_Y_SHIFT = Y_SHIFT = 16
PY_X_SHIFT = X_SHIFT = 32
PY_KEY_MASK = KEY_MASK = BLOCK_KEY_MASK
PY_POSITIVE_MASK = (1 << 63) - 1
PENDING_POOL_IDX = PENDING_POOL_IDX_WP = 65535
MAX_POOL_IDX = DEFAULT_HASH_LAYOUT.max_pool_idx


def resolve_feature_integration_kernel(
    feature_integration_kernel: str, feature_dim: int, support_capacity: int
) -> str:
    if feature_integration_kernel not in FEATURE_INTEGRATION_KERNEL_MODES:
        raise ValueError(f"unknown feature integration kernel: {feature_integration_kernel}")
    if feature_dim < 0 or support_capacity < 1:
        raise ValueError("feature_dim must be nonnegative and support_capacity positive")
    if feature_integration_kernel == "auto":
        return "tiled" if feature_dim > 32 else "grouped"
    return feature_integration_kernel


def validate_grid_shape_for_hash_layout(
    grid_shape, block_size: int, *, layout: HashLayout = DEFAULT_HASH_LAYOUT,
    field_name: str = "grid_shape",
):
    _validate_block_size(block_size)
    if grid_shape is None:
        return None
    if len(grid_shape) != 3 or any(int(v) <= 0 for v in grid_shape):
        raise ValueError(f"{field_name} must contain three positive integers")
    return tuple(int(v) for v in grid_shape)


def _validate_block_size(value: int) -> None:
    if value < 1 or value > 32 or (value & (value - 1)):
        raise ValueError("block_size must be 1 or a power of two through 32")


def _validate_color_grid_size(value: int) -> None:
    if value < 1:
        raise ValueError("color_grid_size must be positive")


def _validate_feature_block_grid_size(value: int) -> None:
    if value < 1:
        raise ValueError("feature_block_grid_size must be positive")


def _validate_feature_channels_per_thread(value: int) -> None:
    if value < 1:
        raise ValueError("feature_channels_per_thread must be positive")


def _validate_feature_grid_shape(*args, **kwargs) -> None:
    return None


def _validate_feature_integration_kernel(value: str) -> None:
    if value not in FEATURE_INTEGRATION_KERNEL_MODES:
        raise ValueError(f"unknown feature integration kernel: {value}")


def _validate_lidar_config(*args, **kwargs) -> None:
    if args and int(args[0]) > 0:
        raise NotImplementedError("lidar mapping is unavailable in the portable mapper")
