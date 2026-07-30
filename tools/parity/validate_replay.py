#!/usr/bin/env python3
"""Validate coverage, hashes, provenance, and fallback/equivalence safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .replay_registry import BY_ID, PIN


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        for side in ("input", "output"):
            path = folder / manifest[side]["file"]
            if sha(path) != manifest[side]["sha256"]:
                raise SystemExit(f"{side} hash mismatch: {row['capability']}")
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    data[key]
        expected = BY_ID[row["capability"]]
        if manifest["tolerance"] != {"rtol": expected.rtol, "atol": expected.atol}:
            raise SystemExit(f"tolerance mismatch: {row['capability']}")
    print(f"validated {len(ids)} deterministic replay cases at pinned upstream {PIN}")


if __name__ == "__main__":
    main()
