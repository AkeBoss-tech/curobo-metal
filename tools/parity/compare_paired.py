#!/usr/bin/env python3
"""Strictly validate and compare pinned CUDA outputs with committed Metal replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_adapters import ADAPTERS
from .replay_registry import BY_ID, PIN


FORMAT = "curobo-metal-paired-replay"
VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _owned_file(folder: Path, name: object) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"unsafe replay filename: {name!r}")
    path = folder / name
    if not path.is_file():
        raise ValueError(f"missing replay file: {path}")
    return path


def _validate_common(manifest: dict[str, Any], capability: str) -> None:
    case = BY_ID[capability]
    expected = {
        "format": FORMAT,
        "version": VERSION,
        "capability": capability,
        "operation": case.operation,
        "upstream_revision": PIN,
        "fallback_enabled": False,
        "equivalence_claimed": False,
        "tolerance": {"rtol": case.rtol, "atol": case.atol},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{capability}: invalid {key}: {manifest.get(key)!r}; expected {value!r}"
            )


def _validate_required_evidence(
    manifest: dict[str, Any], outputs: np.lib.npyio.NpzFile, capability: str
) -> None:
    """Reject a paired report that silently drops invalid or edge coverage."""
    case = BY_ID[capability]
    expected = {
        "invalid": {
            "case": case.invalid_case,
            "output": "invalid_rejected",
            "executed": True,
        },
        "edge": {
            "case": case.edge_case,
            "output": "edge_observed",
            "executed": True,
        },
    }
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or any(
        evidence.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{capability}: required invalid/edge evidence metadata is missing")
    for label, record in expected.items():
        key = record["output"]
        if key not in outputs.files:
            raise ValueError(f"{capability}: committed {label}-case evidence is missing")
        value = outputs[key]
        if value.shape != (1,) or value.dtype != np.int8 or int(value[0]) != 1:
            raise ValueError(f"{capability}: committed {label}-case evidence did not execute")


def compare_capability(
    metal_root: Path, cuda_root: Path, capability: str
) -> dict[str, Any]:
    """Validate provenance and compare every output tensor for one capability."""
    if capability not in ADAPTERS:
        raise ValueError(f"{capability}: no real CUDA adapter is registered")
    case = BY_ID[capability]
    metal_folder = metal_root / capability
    cuda_folder = cuda_root / capability
    metal_manifest = _load_json(metal_folder / "metal-manifest.json")
    cuda_manifest = _load_json(cuda_folder / "cuda-manifest.json")
    _validate_common(metal_manifest, capability)
    _validate_common(cuda_manifest, capability)
    if metal_manifest.get("backend") != "metal" or metal_manifest.get("device") != "mps":
        raise ValueError(f"{capability}: reference side is not fallback-disabled Metal")
    if cuda_manifest.get("backend") != "cuda" or cuda_manifest.get("device") != "cuda":
        raise ValueError(f"{capability}: candidate side is not CUDA")
    if cuda_manifest.get("status") != "complete":
        raise ValueError(f"{capability}: CUDA replay is not complete")
    runtime = cuda_manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("upstream_revision") != PIN
        or not isinstance(runtime.get("torch_cuda"), str)
        or not runtime["torch_cuda"]
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime["cuda_device_name"]
        or runtime.get("cuda_device_count", 0) < 1
        or not (
            isinstance(runtime.get("cuda_capability"), list)
            and len(runtime["cuda_capability"]) == 2
            and all(isinstance(value, int) for value in runtime["cuda_capability"])
        )
    ):
        raise ValueError(f"{capability}: incomplete CUDA runtime provenance")

    input_path = _owned_file(metal_folder, metal_manifest["input"]["file"])
    metal_output = _owned_file(metal_folder, metal_manifest["output"]["file"])
    cuda_output = _owned_file(cuda_folder, cuda_manifest["output"]["file"])
    if sha256(input_path) != metal_manifest["input"]["sha256"]:
        raise ValueError(f"{capability}: committed input hash mismatch")
    if cuda_manifest.get("input_sha256") != sha256(input_path):
        raise ValueError(f"{capability}: CUDA output consumed a different input")
    with np.load(input_path, allow_pickle=False) as inputs:
        if cuda_manifest.get("input_tensor_count") != len(inputs.files):
            raise ValueError(f"{capability}: CUDA input tensor count mismatch")
    if sha256(metal_output) != metal_manifest["output"]["sha256"]:
        raise ValueError(f"{capability}: committed Metal output hash mismatch")
    if sha256(cuda_output) != cuda_manifest["output"]["sha256"]:
        raise ValueError(f"{capability}: CUDA output hash mismatch")

    report: dict[str, Any] = {
        "capability": capability,
        "operation": case.operation,
        "rtol": case.rtol,
        "atol": case.atol,
        "tensors": {},
        "passed": True,
    }
    with (
        np.load(metal_output, allow_pickle=False) as metal,
        np.load(cuda_output, allow_pickle=False) as cuda,
    ):
        if set(metal.files) != set(cuda.files):
            raise ValueError(
                f"{capability}: output keys differ: "
                f"{sorted(set(metal.files) ^ set(cuda.files))}"
            )
        _validate_required_evidence(metal_manifest, metal, capability)
        declared = cuda_manifest["output"].get("tensors")
        actual = {
            key: {"shape": list(cuda[key].shape), "dtype": str(cuda[key].dtype)}
            for key in sorted(cuda.files)
        }
        if declared != actual:
            raise ValueError(f"{capability}: CUDA tensor schema does not match manifest")
        for key in sorted(metal.files):
            left, right = metal[key], cuda[key]
            item: dict[str, Any]
            if left.shape != right.shape or left.dtype != right.dtype:
                item = {
                    "passed": False,
                    "reason": "schema",
                    "metal_shape": list(left.shape),
                    "cuda_shape": list(right.shape),
                    "metal_dtype": str(left.dtype),
                    "cuda_dtype": str(right.dtype),
                }
            elif np.issubdtype(left.dtype, np.number):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                passed = bool(
                    np.allclose(
                        left, right, rtol=case.rtol, atol=case.atol, equal_nan=True
                    )
                )
                item = {
                    "passed": passed,
                    "max_abs": float(delta.max(initial=0)),
                    "mean_abs": float(delta.mean()) if delta.size else 0.0,
                }
            else:
                passed = bool(np.array_equal(left, right))
                item = {"passed": passed, "exact": True}
            report["tensors"][key] = item
            report["passed"] = report["passed"] and item["passed"]
    return report


def compare_ready(metal_root: Path, cuda_root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for capability in sorted(ADAPTERS):
        try:
            results.append(compare_capability(metal_root, cuda_root, capability))
        except ValueError as error:
            errors.append({"capability": capability, "error": str(error)})
    return {
        "format": "curobo-metal-paired-report",
        "version": 1,
        "upstream_revision": PIN,
        "required_capabilities": sorted(ADAPTERS),
        "results": results,
        "errors": errors,
        "passed": not errors
        and len(results) == len(ADAPTERS)
        and all(item["passed"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_ready(args.metal_root, args.cuda_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
