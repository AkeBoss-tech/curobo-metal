#!/usr/bin/env python3
"""Validate coverage, hashes, provenance, and fallback/equivalence safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .replay_corpus import load as load_corpus
from .replay_registry import BY_ID, PIN


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_bit(
    capability: str, output: np.lib.npyio.NpzFile, key: str, label: str
) -> None:
    if key not in output.files:
        raise SystemExit(f"required {label} evidence is missing: {capability}")
    value = output[key]
    if value.shape != (1,) or value.dtype != np.int8 or int(value[0]) != 1:
        raise SystemExit(f"required {label} evidence did not execute: {capability}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    index = json.loads((args.root / "index.json").read_text())
    ids = [row["capability"] for row in index["cases"]]
    if len(ids) != 19 or set(ids) != set(BY_ID) or len(ids) != len(set(ids)):
        raise SystemExit("replay index must cover each of the 19 evidence-blocked capabilities exactly once")
    for row in index["cases"]:
        manifest_path = args.root / row["manifest"]
        if sha(manifest_path) != row["manifest_sha256"]:
            raise SystemExit(f"manifest hash mismatch: {row['capability']}")
        manifest = json.loads(manifest_path.read_text())
        folder = manifest_path.parent
        if manifest["upstream_revision"] != PIN or manifest["fallback_enabled"] or manifest["equivalence_claimed"]:
            raise SystemExit(f"unsafe provenance: {row['capability']}")
        expected = BY_ID[row["capability"]]
        raw, corpus = load_corpus(args.root / "corpus", expected)
        declared_corpus = {
            "case_file": corpus["case_file"].name,
            "case_sha256": corpus["case_sha256"],
            "common_file": corpus["common_file"].name,
            "common_sha256": corpus["common_sha256"],
        }
        if manifest.get("corpus") != declared_corpus:
            raise SystemExit(f"corpus provenance mismatch: {row['capability']}")
        for side in ("input", "output"):
            path = folder / manifest[side]["file"]
            if sha(path) != manifest[side]["sha256"]:
                raise SystemExit(f"{side} hash mismatch: {row['capability']}")
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    data[key]
                if side == "input":
                    expected_keys = set(raw)
                    if set(data.files) != expected_keys:
                        raise SystemExit(f"input schema mismatch: {row['capability']}")
                    for key in expected_keys:
                        if not np.array_equal(data[key], raw[key]):
                            raise SystemExit(f"input corpus mismatch: {row['capability']}:{key}")
                else:
                    evidence = manifest.get("evidence")
                    required = corpus["required_evidence"]
                    expected_evidence = {
                        "invalid": {
                            "case": required["invalid"]["case"],
                            "output": required["invalid"]["output"],
                            "executed": True,
                        },
                        "edge": {
                            "case": required["edge"]["case"],
                            "output": required["edge"]["output"],
                            "executed": True,
                        },
                    }
                    if not isinstance(evidence, dict) or any(
                        evidence.get(key) != value for key, value in expected_evidence.items()
                    ):
                        raise SystemExit(f"required invalid/edge evidence metadata is missing: {row['capability']}")
                    _require_bit(row["capability"], data, required["invalid"]["output"], "invalid-case")
                    _require_bit(row["capability"], data, required["edge"]["output"], "edge-case")
        if manifest["tolerance"] != {"rtol": expected.rtol, "atol": expected.atol}:
            raise SystemExit(f"tolerance mismatch: {row['capability']}")
    print(f"validated {len(ids)} deterministic replay cases at pinned upstream {PIN}")


if __name__ == "__main__":
    main()
