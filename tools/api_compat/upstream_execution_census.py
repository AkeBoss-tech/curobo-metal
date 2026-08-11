#!/usr/bin/env python3
"""Initialize or validate the pinned upstream test/example execution census."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.api_compat.inventory import PINNED_REVISION
except ModuleNotFoundError:  # Direct script execution sets sys.path to this directory.
    _INVENTORY_PATH = Path(__file__).with_name("inventory.py")
    _SPEC = importlib.util.spec_from_file_location("api_compat_inventory_census", _INVENTORY_PATH)
    assert _SPEC and _SPEC.loader
    _inventory_module = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_inventory_module)
    PINNED_REVISION = _inventory_module.PINNED_REVISION


SCHEMA_VERSION = 1
TRACKED_SURFACES = ("bundled_test", "bundled_example")
DISPOSITIONS = (
    "unreviewed",
    "applicable_unchanged",
    "platform_substituted",
    "external_unavailable",
    "not_applicable",
)


def _inventory_entries(inventory: dict[str, Any]) -> list[dict[str, str]]:
    upstream = inventory.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("revision") != PINNED_REVISION:
        raise ValueError("inventory is not pinned to the expected cuRobo V2 revision")
    modules = inventory.get("modules")
    if not isinstance(modules, list):
        raise ValueError("inventory modules must be a list")
    result = []
    for module in modules:
        if module.get("surface") in TRACKED_SURFACES:
            name = module.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("tracked inventory module has no valid name")
            result.append({"module": name, "surface": module["surface"]})
    result.sort(key=lambda item: (item["surface"], item["module"]))
    if len({item["module"] for item in result}) != len(result):
        raise ValueError("inventory contains duplicate tracked module names")
    return result


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_surface = Counter(item["surface"] for item in entries)
    by_disposition = Counter(item["disposition"] for item in entries)
    return {
        "total": len(entries),
        "by_surface": {key: by_surface.get(key, 0) for key in TRACKED_SURFACES},
        "by_disposition": {key: by_disposition.get(key, 0) for key in DISPOSITIONS},
        "review_complete": by_disposition.get("unreviewed", 0) == 0,
    }


def initialize(inventory: dict[str, Any]) -> dict[str, Any]:
    entries = [
        {
            **item,
            "disposition": "unreviewed",
            "rationale": "Pending applicability and dependency review.",
            "evidence": [],
        }
        for item in _inventory_entries(inventory)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": PINNED_REVISION,
        "classification_contract": {
            "unreviewed": "No release claim may be derived until this entry is reviewed.",
            "applicable_unchanged": "The pinned module must run without edits against the installed wheel.",
            "platform_substituted": "The workload is relevant but requires a documented non-CUDA execution substitution.",
            "external_unavailable": "Execution requires an unavailable external ecosystem or service.",
            "not_applicable": "The module does not test or demonstrate the supported release contract.",
        },
        "summary": _summary(entries),
        "entries": entries,
    }


def validate(inventory: dict[str, Any], census: dict[str, Any], *, require_reviewed: bool = False) -> None:
    if census.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported census schema_version: {census.get('schema_version')!r}")
    if census.get("upstream_revision") != PINNED_REVISION:
        raise ValueError("census is not pinned to the expected cuRobo V2 revision")
    entries = census.get("entries")
    if not isinstance(entries, list):
        raise ValueError("census entries must be a list")
    expected = _inventory_entries(inventory)
    observed_identity: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"census entry {index} must be an object")
        module, surface = entry.get("module"), entry.get("surface")
        if not isinstance(module, str) or not module:
            raise ValueError(f"census entry {index} has no valid module")
        if module in seen:
            raise ValueError(f"duplicate census module: {module}")
        seen.add(module)
        if surface not in TRACKED_SURFACES:
            raise ValueError(f"invalid surface for {module}: {surface!r}")
        disposition = entry.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"invalid disposition for {module}: {disposition!r}")
        rationale = entry.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"missing rationale for {module}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError(f"invalid evidence list for {module}")
        if disposition != "unreviewed" and not evidence:
            raise ValueError(f"reviewed entry has no evidence: {module}")
        observed_identity.append({"module": module, "surface": surface})
    observed_identity.sort(key=lambda item: (item["surface"], item["module"]))
    if observed_identity != expected:
        missing = sorted({item["module"] for item in expected} - seen)
        extra = sorted(seen - {item["module"] for item in expected})
        raise ValueError(f"census does not match pinned inventory; missing={missing}, extra={extra}")
    calculated = _summary(entries)
    if census.get("summary") != calculated:
        raise ValueError("census summary is stale or inconsistent with entries")
    if require_reviewed and not calculated["review_complete"]:
        raise ValueError(f"census review incomplete: {calculated['by_disposition']['unreviewed']} unreviewed")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args(argv)
    try:
        inventory = _load(args.inventory)
        if args.initialize:
            if args.require_reviewed:
                raise ValueError("--initialize and --require-reviewed cannot be combined")
            payload = initialize(inventory)
            args.census.parent.mkdir(parents=True, exist_ok=True)
            args.census.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            validate(inventory, _load(args.census), require_reviewed=args.require_reviewed)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
