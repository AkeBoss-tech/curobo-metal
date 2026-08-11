#!/usr/bin/env python3
"""Produce a deterministic, AST-only public export and callable-shape report.

This is evidence about Python declaration shape only.  It intentionally does not
claim numerical, CUDA/Warp ABI, device, autograd, or performance equivalence.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

try:  # Direct ``python tools/api_compat/surface_gate.py`` execution.
    from inventory import PINNED_REVISION, _local_candidates, scan_module
except ModuleNotFoundError:  # Test/importlib execution without mutating sys.path.
    _INVENTORY_PATH = Path(__file__).with_name("inventory.py")
    _SPEC = importlib.util.spec_from_file_location("api_compat_inventory_shared", _INVENTORY_PATH)
    assert _SPEC and _SPEC.loader
    _inventory = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_inventory)
    PINNED_REVISION = _inventory.PINNED_REVISION
    _local_candidates = _inventory._local_candidates
    scan_module = _inventory.scan_module


SCHEMA_VERSION = 1


def _symbols(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Retain one deterministic top-level record for every public name."""
    result: dict[str, dict[str, Any]] = {}
    for item in module["symbols"]:
        # A class/function declaration is more informative than an imported alias
        # with the same spelling, while both still prove the export exists.
        current = result.get(item["name"])
        if current is None or current["kind"] == "reexport":
            result[item["name"]] = item
    return result


def _callable_shape(symbol: dict[str, Any]) -> dict[str, Any] | None:
    if symbol["kind"] == "function":
        return {"kind": "function", "signature": symbol["signature"]}
    if symbol["kind"] == "class":
        return {
            "kind": "class",
            "constructor": symbol["constructor"],
            "methods": {
                item["name"]: item["signature"]
                for item in symbol["members"]
                if item["kind"] == "method"
            },
        }
    return None


def _parse_local(module_name: str, local_root: Path) -> dict[str, Any] | None:
    path = next((path for path in _local_candidates(local_root, module_name) if path.is_file()), None)
    if path is None:
        return None
    # scan_module needs a package directory solely to derive the name/path.  The
    # gate already knows the intended name, so parse the file with the same AST
    # rules and replace those two local bookkeeping fields.
    pseudo_package = local_root
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    del tree  # Syntax validation remains explicit; scan_module performs the scan.
    scanned = scan_module(pseudo_package, path)
    scanned["name"] = module_name
    return scanned


def compare_module(upstream: dict[str, Any], local_root: Path) -> dict[str, Any]:
    """Compare one module's recoverable Python surface without importing it."""
    upstream_symbols = _symbols(upstream)
    local = _parse_local(upstream["name"], local_root)
    if local is None:
        return {
            "name": upstream["name"],
            "module": "missing",
            "exports": {"expected": len(upstream_symbols), "present": 0, "missing": sorted(upstream_symbols)},
            "callables": {"compared": 0, "matching": 0, "different": []},
        }
    local_symbols = _symbols(local)
    missing = sorted(set(upstream_symbols) - set(local_symbols))
    comparable = []
    matching = 0
    different = []
    for name in sorted(set(upstream_symbols) & set(local_symbols)):
        expected = _callable_shape(upstream_symbols[name])
        actual = _callable_shape(local_symbols[name])
        if expected is None or actual is None:
            continue
        comparable.append(name)
        if expected == actual:
            matching += 1
        else:
            different.append(name)
    return {
        "name": upstream["name"],
        "module": "present",
        "exports": {
            "expected": len(upstream_symbols),
            "present": len(set(upstream_symbols) & set(local_symbols)),
            "missing": missing,
        },
        "callables": {"compared": len(comparable), "matching": matching, "different": different},
    }


def build_report(
    payload: dict[str, Any],
    local_root: Path,
    *,
    public_facades_only: bool = False,
) -> dict[str, Any]:
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("revision") != PINNED_REVISION:
        raise ValueError("inventory is not pinned to the expected cuRobo V2 revision")
    modules = payload.get("modules")
    if not isinstance(modules, list):
        raise ValueError("inventory modules must be a list")
    selected = [module for module in modules if module.get("surface") == "runtime"]
    if public_facades_only:
        selected = [module for module in selected if not module.get("name", "").startswith("curobo._src")]
    compared = [compare_module(module, local_root) for module in selected]
    compared.sort(key=lambda item: item["name"])
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream": {"revision": PINNED_REVISION},
        "method": {
            "parser": "python-ast",
            "imports_executed": False,
            "scope": "top_level_exports_and_declared_callable_shapes",
            "module_contract": (
                "runtime_modules_excluding_curobo._src"
                if public_facades_only
                else "all_runtime_modules"
            ),
            "non_claims": [
                "numerical equivalence",
                "CUDA or Warp ABI equivalence",
                "device, autograd, or performance equivalence",
                "dynamic exports or runtime-generated methods",
            ],
        },
        "summary": {
            "modules": len(compared),
            "present_modules": sum(item["module"] == "present" for item in compared),
            "missing_modules": sum(item["module"] == "missing" for item in compared),
            "expected_exports": sum(item["exports"]["expected"] for item in compared),
            "present_exports": sum(item["exports"]["present"] for item in compared),
            "compared_callables": sum(item["callables"]["compared"] for item in compared),
            "matching_callable_shapes": sum(item["callables"]["matching"] for item in compared),
            "different_callable_shapes": sum(len(item["callables"]["different"]) for item in compared),
        },
        "modules": compared,
    }


def _encoded(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("src"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-exact-exports", action="store_true")
    parser.add_argument(
        "--public-facades-only",
        action="store_true",
        help="exclude curobo._src from the scoped alpha facade report",
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(
            json.loads(args.inventory.read_text(encoding="utf-8")),
            args.local_root.resolve(),
            public_facades_only=args.public_facades_only,
        )
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as error:
        parser.error(str(error))
    encoded = _encoded(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.require_exact_exports and (report["summary"]["missing_modules"] or report["summary"]["expected_exports"] != report["summary"]["present_exports"]):
        print("public export gate failed: missing module exports remain", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
