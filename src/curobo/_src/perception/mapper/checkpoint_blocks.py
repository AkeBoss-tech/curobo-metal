"""Portable mapper checkpoint helpers.

These functions preserve the V2 *disk-contract* names while saving ordinary
PyTorch dense tensors.  A native Warp block-pool payload is deliberately
rejected: its hash-table and fp16 packed buffer ABI cannot be reconstructed on
Metal safely.
"""

from __future__ import annotations

import math
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .constants import PY_HASH_PRIME_X, PY_HASH_PRIME_Y, PY_HASH_PRIME_Z, PY_POSITIVE_MASK

BLOCK_CHECKPOINT_FORMAT = "curobo.mapper_blocks"
BLOCK_CHECKPOINT_SCHEMA_VERSION = 1.0
BLOCK_CHECKPOINT_KEYS = {"format", "schema_version", "block_metadata", "blocks"}
BLOCK_METADATA_KEYS = {
    "voxel_size", "block_size", "truncation_distance", "grid_center", "grid_shape",
    "has_dynamic", "has_static", "feature_dim", "feature_block_grid_size", "color_grid_size",
}


def _value(source: Any, name: str, default: Any) -> Any:
    data = getattr(source, "data", source)
    return getattr(data, name, default)


def build_block_metadata(tsdf) -> Dict[str, Any]:
    """Build a source-shaped metadata record for portable dense TSDF state."""
    grid_shape = tuple(int(x) for x in _value(tsdf, "grid_shape", getattr(tsdf, "tsdf", torch.empty(0)).shape[-3:]))
    return {
        "voxel_size": float(_value(tsdf, "voxel_size", 1.0)),
        "block_size": int(_value(tsdf, "block_size", 8)),
        "truncation_distance": float(_value(tsdf, "truncation_distance", 0.04)),
        "grid_center": [float(x) for x in torch.as_tensor(_value(tsdf, "origin", (0.0, 0.0, 0.0))).reshape(-1).tolist()[:3]],
        "grid_shape": list(grid_shape),
        "has_dynamic": True,
        "has_static": False,
        "feature_dim": int(_value(tsdf, "feature_dim", 0)),
        "feature_block_grid_size": int(_value(tsdf, "feature_block_grid_size", 1)),
        "color_grid_size": int(_value(tsdf, "color_grid_size", 1)),
    }


def save_block_checkpoint(file_path: Union[str, PathLike[str]], block_metadata: Dict[str, Any],
                          blocks: Dict[str, torch.Tensor]) -> None:
    checkpoint = {
        "format": BLOCK_CHECKPOINT_FORMAT,
        "schema_version": BLOCK_CHECKPOINT_SCHEMA_VERSION,
        "block_metadata": dict(block_metadata),
        "blocks": clone_blocks_to_cpu(blocks),
    }
    validate_block_checkpoint(checkpoint)
    torch.save(checkpoint, file_path)


def load_block_checkpoint(file_path: Union[str, PathLike[str]]) -> Dict[str, Any]:
    kwargs = {"map_location": "cpu"}
    if "weights_only" in torch.load.__code__.co_varnames:
        kwargs["weights_only"] = True
    checkpoint = torch.load(file_path, **kwargs)
    validate_block_checkpoint(checkpoint)
    return checkpoint


def clone_blocks_to_cpu(blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: require_tensor(blocks, key).detach().cpu().clone() for key in blocks}


def clone_blocks(blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: require_tensor(blocks, key).detach().clone() for key in blocks}


def validate_block_checkpoint(checkpoint: Any) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("block checkpoint must be a dictionary")
    if checkpoint.get("format") != BLOCK_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported block checkpoint format {checkpoint.get('format')!r}")
    if checkpoint.get("schema_version") != BLOCK_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported block checkpoint schema version")
    if set(checkpoint) != BLOCK_CHECKPOINT_KEYS:
        raise ValueError("checkpoint must contain exactly format, schema_version, block_metadata, and blocks")
    validate_block_metadata(checkpoint["block_metadata"])
    validate_block_payload(checkpoint["blocks"], checkpoint["block_metadata"])


def validate_block_metadata(block_metadata: Any) -> None:
    if not isinstance(block_metadata, dict):
        raise ValueError("block_metadata must be a dictionary")
    if set(block_metadata) != BLOCK_METADATA_KEYS:
        raise ValueError("block_metadata fields do not match the portable checkpoint schema")
    require_positive_float(block_metadata, "voxel_size")
    require_positive_float(block_metadata, "truncation_distance")
    require_positive_int(block_metadata, "block_size")
    require_positive_int(block_metadata, "feature_block_grid_size")
    require_positive_int(block_metadata, "color_grid_size")
    require_nonnegative_int(block_metadata, "feature_dim")
    require_bool(block_metadata, "has_dynamic")
    require_bool(block_metadata, "has_static")
    require_vec3(block_metadata, "grid_center")
    require_int3(block_metadata, "grid_shape")
    if not block_metadata["has_dynamic"]:
        raise ValueError("portable dense checkpoints require dynamic TSDF data")


def validate_block_payload(blocks: Any, block_metadata: Dict[str, Any]) -> None:
    del block_metadata
    if not isinstance(blocks, dict) or not blocks:
        raise ValueError("blocks must be a nonempty dictionary of tensors")
    for key in blocks:
        require_tensor(blocks, key)


def validate_block_metadata_for_target(block_metadata: Dict[str, Any], tsdf) -> None:
    validate_block_metadata(block_metadata)
    target = build_block_metadata(tsdf)
    for field in ("block_size", "grid_shape", "feature_dim", "feature_block_grid_size", "color_grid_size"):
        if block_metadata[field] != target[field]:
            raise ValueError(f"{field} mismatch: checkpoint={block_metadata[field]!r}, target={target[field]!r}")
    for field in ("voxel_size", "truncation_distance"):
        require_close(float(block_metadata[field]), float(target[field]), field)


def ceil_div_positive(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("value must be nonnegative and divisor positive")
    return (value + divisor - 1) // divisor


def signed_int64_from_uint64(value: int) -> int:
    value = int(value) & ((1 << 64) - 1)
    return value - (1 << 64) if value >= (1 << 63) else value


def pack_hash_entry_host(bx: int, by: int, bz: int, pool_idx: int) -> int:
    """Stable portable host hash packing; this is not a Warp ABI promise."""
    for coordinate in (bx, by, bz):
        if not -(1 << 15) <= int(coordinate) < (1 << 15):
            raise ValueError("block coordinate is outside the portable int16 hash range")
    if not 0 <= int(pool_idx) < (1 << 16):
        raise ValueError("pool_idx is outside the portable uint16 hash range")
    packed = (((int(bx) + (1 << 15)) & 0xFFFF) << 48) | (((int(by) + (1 << 15)) & 0xFFFF) << 32) | (((int(bz) + (1 << 15)) & 0xFFFF) << 16) | int(pool_idx)
    return signed_int64_from_uint64(packed)


def spatial_hash_host(bx: int, by: int, bz: int, capacity: int) -> int:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return int((((int(bx) * PY_HASH_PRIME_X) ^ (int(by) * PY_HASH_PRIME_Y) ^ (int(bz) * PY_HASH_PRIME_Z)) & PY_POSITIVE_MASK) % capacity)


def validate_import_block_coords_unique(coords: torch.Tensor) -> None:
    if coords.ndim != 2 or coords.shape[-1] != 3 or coords.dtype not in (torch.int32, torch.int64):
        raise ValueError("block coordinates must be an int32/int64 [N, 3] tensor")
    if coords.shape[0] and torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
        raise ValueError("block coordinates must be unique")


def validate_import_block_coords_for_hash_layout(coords: torch.Tensor) -> None:
    validate_import_block_coords_unique(coords)
    if coords.numel() and bool((coords.abs() >= (1 << 15)).any().item()):
        raise ValueError("block coordinates are outside the portable int16 hash range")


def validate_import_block_coords_for_grid(coords: torch.Tensor, grid_shape: Tuple[int, int, int], block_size: int) -> None:
    validate_import_block_coords_unique(coords)
    blocks = torch.tensor([ceil_div_positive(int(n), int(block_size)) for n in grid_shape], device=coords.device)
    if coords.numel() and bool(((coords < 0) | (coords >= blocks)).any().item()):
        raise ValueError("block coordinates lie outside grid_shape")


def rebuild_import_hash_state(coords: torch.Tensor, hash_capacity: int, max_blocks: int):
    validate_import_block_coords_for_hash_layout(coords)
    if coords.shape[0] > max_blocks:
        raise ValueError("import contains more blocks than max_blocks")
    hashes = torch.full((hash_capacity,), -1, dtype=torch.int64, device=coords.device)
    for index, xyz in enumerate(coords.tolist()):
        slot = spatial_hash_host(*xyz, hash_capacity)
        while hashes[slot] >= 0:
            slot = (slot + 1) % hash_capacity
        hashes[slot] = index
    return hashes


def apply_constant_dynamic_weight(blocks: Dict[str, torch.Tensor], import_weight: float):
    for key in ("weight", "block_grid_weight"):
        if key in blocks:
            blocks[key] = torch.full_like(blocks[key], import_weight)
    return blocks


def apply_constant_feature_weight(blocks: Dict[str, torch.Tensor], import_weight: float):
    return apply_constant_dynamic_weight(blocks, import_weight)


def prepare_blocks_for_import(blocks: Dict[str, torch.Tensor], block_metadata: Dict[str, Any], *,
                              import_weight: Optional[float], minimum_tsdf_weight: float,
                              block_empty_threshold: float):
    validate_block_metadata(block_metadata)
    result = clone_blocks(blocks)
    if import_weight is not None:
        if import_weight <= 0:
            raise ValueError("import_weight must be positive")
        apply_constant_dynamic_weight(result, import_weight)
        apply_constant_feature_weight(result, import_weight)
    validate_recycle_threshold(result, block_metadata, block_empty_threshold)
    if minimum_tsdf_weight < 0:
        raise ValueError("minimum_tsdf_weight must be nonnegative")
    return result


def validate_recycle_threshold(blocks: Dict[str, torch.Tensor], block_metadata: Dict[str, Any], block_empty_threshold: float):
    validate_block_payload(blocks, block_metadata)
    if block_empty_threshold < 0:
        raise ValueError("block_empty_threshold must be nonnegative")


def require_tensor(blocks: Dict[str, torch.Tensor], key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = blocks.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"block payload field {key!r} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"block payload field {key!r} must have dtype {dtype}")
    return value


def require_shape(tensor: torch.Tensor, key: str, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{key} must have shape {shape}, got {tuple(tensor.shape)}")


def require_positive_float(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), (int, float)) or not math.isfinite(float(values[key])) or values[key] <= 0:
        raise ValueError(f"{key} must be a positive finite float")


def require_positive_int(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), int) or isinstance(values[key], bool) or values[key] <= 0:
        raise ValueError(f"{key} must be a positive integer")


def require_nonnegative_int(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), int) or isinstance(values[key], bool) or values[key] < 0:
        raise ValueError(f"{key} must be a nonnegative integer")


def require_bool(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), bool):
        raise ValueError(f"{key} must be a bool")


def require_vec3(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), (list, tuple)) or len(values[key]) != 3:
        raise ValueError(f"{key} must be a length-3 vector")


def require_int3(values: Dict[str, Any], key: str) -> None:
    if not isinstance(values.get(key), (list, tuple)) or len(values[key]) != 3 or any(not isinstance(x, int) or x <= 0 for x in values[key]):
        raise ValueError(f"{key} must be three positive integers")


def require_close(source: float, target: float, field_name: str) -> None:
    if not math.isclose(source, target, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{field_name} mismatch: checkpoint={source}, target={target}")
