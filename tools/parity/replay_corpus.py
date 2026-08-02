"""Read the checked-in, capability-owned replay input corpus.

Inputs are deliberately data rather than Python literals in ``generate_replay``.
That lets a CUDA runner verify exactly which case material it consumed and makes
reviewing a capability's normal, invalid, and edge coverage straightforward.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .replay_registry import Case


FORMAT = "curobo-metal-replay-corpus"
VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON corpus {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"corpus {path} must be a JSON object")
    return value


def _tensor(name: str, value: Any) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"corpus tensor {name} must be an object")
    if value.get("encoding") == "utf8":
        text = value.get("value")
        if not isinstance(text, str):
            raise ValueError(f"UTF-8 corpus tensor {name} must contain text")
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
    dtype = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not all(
        isinstance(axis, int) and axis >= 0 for axis in shape
    ):
        raise ValueError(f"corpus tensor {name} has invalid dtype or shape")
    try:
        result = np.asarray(value.get("data"), dtype=np.dtype(dtype))
    except (TypeError, ValueError) as error:
        raise ValueError(f"corpus tensor {name} has invalid data") from error
    target_shape = tuple(shape)
    if result.size == int(np.prod(target_shape, dtype=np.int64)) and tuple(result.shape) != target_shape:
        result = result.reshape(target_shape)
    if tuple(result.shape) != target_shape:
        raise ValueError(
            f"corpus tensor {name} has shape {tuple(result.shape)}, expected {target_shape}"
        )
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise ValueError(f"corpus tensor {name} must be finite")
    return result


def load(corpus_root: Path, case: Case) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one deterministic case and validate its declared evidence contract."""
    shared_path = corpus_root / "common.json"
    case_path = corpus_root / f"{case.capability}.json"
    shared = _json_object(shared_path)
    spec = _json_object(case_path)
    if shared.get("format") != FORMAT or shared.get("version") != VERSION:
        raise ValueError("invalid common replay corpus format")
    if spec.get("format") != FORMAT or spec.get("version") != VERSION:
        raise ValueError(f"invalid replay corpus format: {case.capability}")
    if spec.get("capability") != case.capability:
        raise ValueError(f"corpus capability mismatch: {case.capability}")
    keys = spec.get("inputs")
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
        raise ValueError(f"corpus inputs must be a nonempty string list: {case.capability}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"corpus input keys must be unique: {case.capability}")
    common_tensors = shared.get("tensors")
    if not isinstance(common_tensors, dict):
        raise ValueError("common replay corpus has no tensors object")
    missing = sorted(set(keys) - set(common_tensors))
    if missing:
        raise ValueError(f"corpus references unknown tensor(s) {missing}: {case.capability}")
    evidence = spec.get("required_evidence")
    expected_evidence = {
        "invalid": {"case": case.invalid_case, "output": "invalid_rejected"},
        "edge": {"case": case.edge_case, "output": "edge_observed"},
    }
    if evidence != expected_evidence:
        raise ValueError(f"corpus evidence contract mismatch: {case.capability}")
    return ({key: _tensor(key, common_tensors[key]) for key in keys}, {
        "case_file": case_path,
        "case_sha256": sha256(case_path),
        "common_file": shared_path,
        "common_sha256": sha256(shared_path),
        "required_evidence": expected_evidence,
    })
