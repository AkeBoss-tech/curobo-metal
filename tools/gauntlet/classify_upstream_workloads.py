#!/usr/bin/env python3
"""Classify obvious pinned-workload boundaries from inspectable source evidence.

This tool deliberately leaves CPU-looking tests unreviewed: only actual clean-wheel
execution can promote those modules to ``applicable_unchanged``.  It does close
package/support files and workloads with explicit platform or external dependencies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.api_compat.upstream_execution_census import PINNED_REVISION, _summary, validate


EXTERNAL_ROOTS = {
    "isaacsim",
    "omni",
    "rclpy",
    "rospy",
    "viser",
    "nvblox_torch",
    "pinocchio",
    "pxr",
    "soma_retargeter",
    "trimesh",
    "yourdfpy",
}
PLATFORM_ROOTS = {"warp"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attribute_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def inspect_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    attributes: set[str] = set()
    cuda_literals: set[str] = set()
    test_declarations: list[str] = []
    cuda_identifiers: set[str] = set()
    has_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.partition(".")[0])
        elif isinstance(node, ast.Attribute):
            name = _attribute_name(node)
            if name:
                attributes.add(name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.lower()
            if value == "cuda" or value.startswith("cuda:"):
                cuda_literals.add(node.value)
        elif isinstance(node, ast.Name) and "cuda" in node.id.lower():
            cuda_identifiers.add(node.id)
        elif isinstance(node, ast.arg) and "cuda" in node.arg.lower():
            cuda_identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            test_declarations.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            test_declarations.append(node.name)
        elif isinstance(node, ast.Compare):
            has_main = has_main or any(
                isinstance(item, ast.Constant) and item.value == "__main__"
                for item in [node.left, *node.comparators]
            )
    external = sorted(imports & EXTERNAL_ROOTS)
    platform = sorted(imports & PLATFORM_ROOTS)
    cuda_attributes = sorted(
        name for name in attributes if name == "torch.cuda" or name.startswith("torch.cuda.")
    )
    if cuda_literals or cuda_attributes or cuda_identifiers:
        platform.append("cuda")
    return {
        "sha256": _sha256(path),
        "imports": sorted(imports),
        "test_declarations": sorted(test_declarations),
        "has_main": has_main,
        "external_dependencies": external,
        "platform_dependencies": sorted(set(platform)),
        "cuda_literals": sorted(cuda_literals),
        "cuda_attributes": cuda_attributes,
        "cuda_identifiers": sorted(cuda_identifiers),
    }


def classify(
    inventory: dict[str, Any], census: dict[str, Any], upstream: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"expected upstream {PINNED_REVISION}, found {revision}")
    validate(inventory, census)
    inventory_by_name = {item["name"]: item for item in inventory["modules"]}
    report_entries: list[dict[str, Any]] = []
    changed = json.loads(json.dumps(census))
    for entry in changed["entries"]:
        module = inventory_by_name[entry["module"]]
        path = upstream / module["path"]
        evidence = inspect_source(path)
        if entry["module"] == "curobo.tests.test_examples":
            evidence["platform_dependencies"] = ["subprocess CUDA examples"]
        if evidence["sha256"] != module["sha256"]:
            raise ValueError(f"source hash differs from inventory for {entry['module']}")
        record = {"module": entry["module"], "path": module["path"], **evidence}
        report_entries.append(record)
        if entry["disposition"] != "unreviewed":
            continue
        is_support = (
            not evidence["test_declarations"]
            and not evidence["has_main"]
            and (
                path.name in {"__init__.py", "conftest.py"}
                or path.stem.endswith("_reference")
            )
        )
        source_fact = (
            f"artifacts/gauntlet/upstream-workload-dependencies.json records pinned source "
            f"{module['path']} at SHA-256 {evidence['sha256']}."
        )
        if is_support:
            entry.update(
                disposition="not_applicable",
                rationale="Support/package module with no independently collected test or executable entry point.",
                evidence=[source_fact],
            )
        elif evidence["external_dependencies"]:
            names = ", ".join(evidence["external_dependencies"])
            entry.update(
                disposition="external_unavailable",
                rationale=f"Pinned workload imports external integration dependencies: {names}.",
                evidence=[source_fact],
            )
        elif evidence["platform_dependencies"]:
            names = ", ".join(evidence["platform_dependencies"])
            entry.update(
                disposition="platform_substituted",
                rationale=(
                    "Relevant workload directly selects platform-specific execution "
                    f"({names}); unchanged Apple-Silicon execution is not claimed."
                ),
                evidence=[source_fact, "gauntlet/curobo-consistency.json records the explicit platform exclusions."],
            )
    changed["summary"] = _summary(changed["entries"])
    validate(inventory, changed)
    report = {
        "schema_version": 1,
        "upstream_revision": revision,
        "method": "Python AST imports, CUDA attributes/literals, test declarations, and entry points",
        "non_claims": [
            "CPU-looking tests pass unchanged",
            "platform-substituted behavior is numerically equivalent",
            "imports hidden behind dynamic loading are exhaustively detected",
        ],
        "summary": {
            "inspected": len(report_entries),
            "external": sum(bool(item["external_dependencies"]) for item in report_entries),
            "platform": sum(bool(item["platform_dependencies"]) for item in report_entries),
            "support": sum(
                (
                    not item["test_declarations"]
                    and not item["has_main"]
                    and (
                        Path(item["path"]).name in {"__init__.py", "conftest.py"}
                        or Path(item["path"]).stem.endswith("_reference")
                    )
                )
                for item in report_entries
            ),
        },
        "entries": report_entries,
    }
    return changed, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/api_compat/upstream-api.json"))
    parser.add_argument("--census", type=Path, default=Path("artifacts/api_compat/upstream-execution-census.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/gauntlet/upstream-workload-dependencies.json"))
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    census = json.loads(args.census.read_text(encoding="utf-8"))
    changed, report = classify(inventory, census, args.upstream.resolve())
    args.census.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(changed["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
