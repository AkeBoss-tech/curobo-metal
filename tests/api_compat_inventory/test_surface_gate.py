from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools/api_compat/surface_gate.py"
SPEC = importlib.util.spec_from_file_location("api_compat_surface_gate", MODULE_PATH)
assert SPEC and SPEC.loader
surface_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surface_gate)


def _inventory(symbols: list[dict]) -> dict:
    return {
        "upstream": {"revision": surface_gate.PINNED_REVISION},
        "modules": [{"name": "curobo.example", "surface": "runtime", "symbols": symbols}],
    }


def test_surface_gate_reports_exports_and_callable_shape(tmp_path: Path) -> None:
    local = tmp_path / "src/curobo"
    local.mkdir(parents=True)
    (local / "example.py").write_text(
        "def calculate(value, *, enabled=True):\n    return value\n\nclass Config:\n    def __init__(self, size=2): pass\n",
        encoding="utf-8",
    )
    symbols = [
        {"name": "calculate", "kind": "function", "signature": {"parameters": [{"name": "value"}, {"name": "enabled"}], "returns": None, "async": False}, "decorators": []},
        {"name": "Config", "kind": "class", "constructor": {"parameters": [{"name": "self"}, {"name": "size"}], "returns": None, "async": False}, "members": []},
        {"name": "Absent", "kind": "assignment"},
    ]
    report = surface_gate.build_report(_inventory(symbols), tmp_path / "src")
    row = report["modules"][0]
    assert row["exports"] == {"expected": 3, "present": 2, "missing": ["Absent"]}
    # Signatures compare complete AST records, not merely parameter counts.
    assert row["callables"]["compared"] == 2
    assert row["callables"]["matching"] == 0
    assert row["callables"]["different"] == ["Config", "calculate"]
    assert report["method"]["imports_executed"] is False


def test_public_facade_scope_excludes_private_src(tmp_path: Path) -> None:
    package = tmp_path / "src/curobo"
    (package / "_src").mkdir(parents=True)
    (package / "public.py").write_text("value = 1\n", encoding="utf-8")
    (package / "_src/private.py").write_text("value = 1\n", encoding="utf-8")
    inventory = {
        "upstream": {"revision": surface_gate.PINNED_REVISION},
        "modules": [
            {"name": "curobo.public", "surface": "runtime", "symbols": [{"name": "value", "kind": "assignment"}]},
            {"name": "curobo._src.private", "surface": "runtime", "symbols": [{"name": "missing", "kind": "assignment"}]},
        ],
    }
    full = surface_gate.build_report(inventory, tmp_path / "src")
    scoped = surface_gate.build_report(
        inventory, tmp_path / "src", public_facades_only=True
    )
    assert full["summary"]["modules"] == 2
    assert scoped["summary"]["modules"] == 1
    assert scoped["summary"]["expected_exports"] == scoped["summary"]["present_exports"] == 1
    assert scoped["method"]["module_contract"] == "runtime_modules_excluding_curobo._src"


def test_callable_replaced_by_noncallable_is_a_shape_difference(tmp_path: Path) -> None:
    local = tmp_path / "src/curobo"
    local.mkdir(parents=True)
    (local / "example.py").write_text("calculate = None\n", encoding="utf-8")
    inventory = _inventory(
        [
            {
                "name": "calculate",
                "kind": "function",
                "signature": {
                    "parameters": [{"name": "value"}],
                    "returns": None,
                    "async": False,
                },
                "decorators": [],
            }
        ]
    )
    row = surface_gate.build_report(inventory, tmp_path / "src")["modules"][0]
    assert row["exports"] == {"expected": 1, "present": 1, "missing": []}
    assert row["callables"] == {
        "compared": 1,
        "matching": 0,
        "different": ["calculate"],
    }
