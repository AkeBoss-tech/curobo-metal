#!/usr/bin/env python3
"""Pinned-upstream CUDA replay runner.

This runner refuses the wrong checkout before importing upstream.  Cases with
no clean asset-independent adapter emit a machine-readable external constraint,
never synthetic numerical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import numpy as np

from .cuda_adapters import ADAPTERS, run
from .replay_registry import BY_ID, PIN


def revision(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--capability", choices=sorted(BY_ID), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = revision(args.upstream)
    if actual != PIN:
        raise SystemExit(f"refusing upstream revision {actual}; required {PIN}")
    with np.load(args.input, allow_pickle=False) as archive:
        # Force complete decoding and prove the exact portable input was consumed.
        tensors = {key: archive[key] for key in archive.files}
    case = BY_ID[args.capability]
    args.output.mkdir(parents=True, exist_ok=True)
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.capability in ADAPTERS:
        # Import the exact checkout only after its revision has been verified.
        sys.path.insert(0, str(args.upstream))
        outputs = run(args.capability, tensors)
        output_path = args.output / "cuda-outputs.npz"
        np.savez(output_path, **outputs)
        output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        record = {
            "format": "curobo-metal-paired-replay", "version": 1,
            "capability": case.capability, "operation": case.operation,
            "backend": "cuda", "device": "cuda", "fallback_enabled": False,
            "upstream_revision": actual, "input_sha256": input_sha,
            "input_tensor_count": len(tensors), "status": "complete",
            "output": {
                "file": output_path.name,
                "sha256": output_sha,
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in sorted(outputs.items())
                },
            },
            "equivalence_claimed": False,
            "tolerance": {"rtol": case.rtol, "atol": case.atol},
        }
        (args.output / "cuda-manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(record, sort_keys=True))
        return
    record = {
        "format": "curobo-metal-paired-replay", "version": 1,
        "capability": case.capability, "operation": case.operation,
        "backend": "cuda", "device": "cuda", "fallback_enabled": False,
        "upstream_revision": actual, "input_sha256": input_sha,
        "input_tensor_count": len(tensors), "status": "external_constraint",
        "constraint": case.cuda_constraint, "equivalence_claimed": False,
        "tolerance": {"rtol": case.rtol, "atol": case.atol},
    }
    (args.output / "cuda-manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
