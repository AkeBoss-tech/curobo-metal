from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[2] / "tools/api_compat/inventory.py"
SPEC = importlib.util.spec_from_file_location("api_compat_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def test_wrong_revision_fails_closed(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "README").write_text("wrong\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "wrong"], check=True)
    with pytest.raises(ValueError, match="wrong upstream revision"):
        inventory.verify_revision(tmp_path)


def test_static_scan_records_signatures_dataclasses_enums_and_reexports(tmp_path: Path) -> None:
    package = tmp_path / "curobo"
    package.mkdir()
    module = package / "sample.py"
    module.write_text(
        """
from elsewhere import PublicThing, hidden as _hidden
from dataclasses import dataclass
from enum import Enum

DEFAULT = 3

@dataclass(frozen=True)
class Config:
    size: int = 2
    def run(self, value: str = "x", *, enabled: bool = True) -> str:
        return value

class Mode(Enum):
    FAST = "fast"

def compute(a: int, /, b=DEFAULT, *args, flag=False, **kwargs) -> bool:
    return True
""",
        encoding="utf-8",
    )
    result = inventory.scan_module(package, module)
    by_name = {symbol["name"]: symbol for symbol in result["symbols"]}
    assert by_name["Config"]["dataclass"] is True
    assert by_name["Config"]["dataclass_options"] == "dataclass(frozen=True)"
    assert by_name["Mode"]["enum"] is True
    assert by_name["Mode"]["members"][0]["kind"] == "enum_member"
    assert by_name["PublicThing"]["kind"] == "reexport"
    parameters = by_name["compute"]["signature"]["parameters"]
    assert [(item["name"], item["kind"], item["default"]) for item in parameters] == [
        ("a", "positional_only", None),
        ("b", "positional_or_keyword", "DEFAULT"),
        ("args", "var_positional", None),
        ("flag", "keyword_only", "False"),
        ("kwargs", "var_keyword", None),
    ]


def test_scanner_never_imports_examined_source(tmp_path: Path) -> None:
    package = tmp_path / "curobo"
    package.mkdir()
    marker = tmp_path / "imported"
    module = package / "danger.py"
    module.write_text(
        f'Path({str(marker)!r}).write_text("bad")\nimport cuda\nPUBLIC = 1\n',
        encoding="utf-8",
    )
    result = inventory.scan_module(package, module)
    assert not marker.exists()
    assert any(item["name"] == "PUBLIC" for item in result["symbols"])


def test_encoding_and_module_order_are_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "upstream/curobo"
    package.mkdir(parents=True)
    (package / "z.py").write_text("Z = 1\n", encoding="utf-8")
    (package / "a.py").write_text("A = 1\n", encoding="utf-8")
    monkeypatch.setattr(inventory, "verify_revision", lambda source: None)
    first = inventory.build_inventory(package.parent, tmp_path / "local")
    second = inventory.build_inventory(package.parent, tmp_path / "local")
    assert inventory._encoded(first) == inventory._encoded(second)
    assert [item["name"] for item in first["modules"]] == ["curobo.a", "curobo.z"]
    json.loads(inventory._encoded(first))


def test_local_classification_is_ast_only(tmp_path: Path) -> None:
    local = tmp_path / "src/curobo_metal"
    local.mkdir(parents=True)
    (local / "thing.py").write_text("import cuda\nclass Present: pass\n", encoding="utf-8")
    module = {
        "name": "curobo.thing",
        "symbols": [{"name": "Present"}, {"name": "Absent"}],
    }
    result = inventory.classify_local(module, tmp_path / "src")
    assert result == {
        "status": "partial",
        "path": "curobo_metal/thing.py",
        "resolved_symbols": 1,
        "missing_symbols": 1,
    }
