"""Keep the vendored CUDA test suite byte-for-byte tied to the compatibility pin."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.upstream.manage import PINNED_SHA


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "src" / "curobo" / "tests"
MANIFEST = SNAPSHOT / "MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_upstream_test_snapshot_is_complete_and_unmodified() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream"]["revision"] == PINNED_SHA
    expected = manifest["files"]
    actual = {
        relative: _sha256(SNAPSHOT / relative)
        for relative in expected
    }
    assert actual == expected
    assert manifest["test_modules"] == 177
