"""Runtime namespace import gate for the pinned cuRobo V2 inventory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


INVENTORY = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "api_compat"
    / "upstream-api.json"
)
RUNTIME_IMPORTS_PATH = Path(__file__).resolve().parents[2] / "tools" / "api_compat" / "runtime_imports.py"
SPEC = importlib.util.spec_from_file_location("api_compat_runtime_imports", RUNTIME_IMPORTS_PATH)
assert SPEC and SPEC.loader
runtime_imports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_imports)


def test_every_pinned_runtime_module_imports_without_optional_backends() -> None:
    """Optional CUDA/Warp/USD dependencies must fail only when their APIs are used."""
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    runtime_modules = runtime_imports.runtime_module_names(payload)
    assert not runtime_imports.audit_runtime_imports(runtime_modules)


def test_runtime_gate_fails_closed_on_invalid_inventory() -> None:
    with pytest.raises(ValueError, match="not pinned"):
        runtime_imports.runtime_module_names({"upstream": {"revision": "wrong"}})


def test_runtime_gate_collects_all_import_failures() -> None:
    failures = runtime_imports.audit_runtime_imports(
        ["curobo.ok", "curobo.bad"],
        importer=lambda name: (_ for _ in ()).throw(ImportError("nope")) if name.endswith("bad") else object(),
    )
    assert failures == {"curobo.bad": "ImportError: nope"}
