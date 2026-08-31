#!/usr/bin/env python3
"""Vendor the pinned cuRobo test tree and record its exact source hashes.

The snapshot is intentionally kept under ``tests/upstream``.  It is a diagnostic
suite for Metal-port parity: failures and collection errors are expected until
the corresponding CUDA behavior is implemented or deliberately classified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.upstream.manage import PINNED_REPOSITORY, PINNED_SHA, verify_pin


DESTINATION = ROOT / "src" / "curobo" / "tests"
MANIFEST = DESTINATION / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(source: Path, destination: Path = DESTINATION) -> dict[str, object]:
    """Copy every pinned test-support file and return its provenance manifest."""
    source = source.resolve()
    verify_pin(source)
    test_root = source / "curobo" / "tests"
    if not test_root.is_dir():
        raise SystemExit(f"upstream test directory is missing: {test_root}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(test_root, destination)
    files = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "upstream": {"repository": PINNED_REPOSITORY, "revision": PINNED_SHA},
        "source_root": "curobo/tests",
        "files": files,
        "test_modules": sum(path.startswith("test_") or "/test_" in path for path in files),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Pinned cuRobo checkout")
    args = parser.parse_args()
    manifest = snapshot(args.source)
    print(f"vendored {manifest['test_modules']} upstream test modules into {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
