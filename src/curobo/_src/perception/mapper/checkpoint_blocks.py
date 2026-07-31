"""Safe portable mapper checkpoint helpers."""

from __future__ import annotations

import os
from typing import Any

import torch

BLOCK_CHECKPOINT_FORMAT = "curobo-metal-blocks"
BLOCK_CHECKPOINT_SCHEMA_VERSION = 1
BLOCK_CHECKPOINT_KEYS = ("format", "schema_version", "metadata", "blocks")
BLOCK_METADATA_KEYS = ("voxel_size", "block_size", "grid_shape")


def clone_blocks(blocks):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in blocks.items()}


def clone_blocks_to_cpu(blocks):
    return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in blocks.items()}


def build_block_metadata(tsdf):
    return {
        "voxel_size": float(getattr(tsdf, "voxel_size", 1.0)),
        "block_size": int(getattr(tsdf, "block_size", 8)),
        "grid_shape": tuple(getattr(tsdf, "grid_shape", ())),
    }


def save_block_checkpoint(file_path, block_metadata, blocks):
    torch.save({
        "format": BLOCK_CHECKPOINT_FORMAT,
        "schema_version": BLOCK_CHECKPOINT_SCHEMA_VERSION,
        "metadata": dict(block_metadata),
        "blocks": clone_blocks_to_cpu(blocks),
    }, file_path)


def load_block_checkpoint(file_path):
    result = torch.load(file_path, map_location="cpu", weights_only=True)
    validate_block_checkpoint(result)
    return result["metadata"], result["blocks"]


def validate_block_checkpoint(checkpoint: Any):
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != BLOCK_CHECKPOINT_FORMAT:
        raise ValueError("invalid portable mapper checkpoint format")
    if checkpoint.get("schema_version") != BLOCK_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported mapper checkpoint schema")
    validate_block_metadata(checkpoint.get("metadata"))
    if not isinstance(checkpoint.get("blocks"), dict):
        raise ValueError("checkpoint blocks must be a dictionary")
    return checkpoint


def validate_block_metadata(value):
    if not isinstance(value, dict):
        raise ValueError("block metadata must be a dictionary")
    return value


def ceil_div_positive(value, divisor):
    if value < 0 or divisor <= 0:
        raise ValueError("value must be nonnegative and divisor positive")
    return (value + divisor - 1) // divisor


def signed_int64_from_uint64(value):
    return value - (1 << 64) if value >= (1 << 63) else value


def apply_constant_dynamic_weight(blocks, import_weight):
    for key in ("weight", "block_grid_weight"):
        if key in blocks:
            blocks[key] = torch.full_like(blocks[key], import_weight)
    return blocks


apply_constant_feature_weight = apply_constant_dynamic_weight
