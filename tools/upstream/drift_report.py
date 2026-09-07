#!/usr/bin/env python3
"""Describe compatibility-relevant drift after the pinned cuRobo revision."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def symbols(source: Path, revision: str, path: str) -> set[str]:
    try:
        value = git(source, "show", f"{revision}:{path}")
        tree = ast.parse(value, filename=path)
    except (subprocess.CalledProcessError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if not name.startswith("_"):
                    found.add(name)
    return found


def category(path: str) -> str:
    if path.startswith("curobo/examples/"):
        return "examples"
    if path.startswith("curobo/tests/"):
        return "behavior_tests"
    if path.startswith("curobo/") and path.endswith(".py"):
        return "python_api"
    if path.startswith("docs/"):
        return "documentation"
    return "packaging_or_other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", default=PINNED_REVISION)
    parser.add_argument("--target", default="origin/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    base, target = git(source, "rev-parse", args.base), git(source, "rev-parse", args.target)
    entries: list[dict] = []
    buckets: dict[str, int] = {}
    changed = git(source, "diff", "--name-status", "--find-renames", base, target)
    for line in changed.splitlines():
        fields = line.split("\t")
        status, path = fields[0], fields[-1]
        group = category(path)
        buckets[group] = buckets.get(group, 0) + 1
        record = {"status": status, "path": path, "category": group}
        if group == "python_api":
            before, after = symbols(source, base, path), symbols(source, target, path)
            record["added_symbols"] = sorted(after - before)
            record["removed_symbols"] = sorted(before - after)
        entries.append(record)
    latest_tag = ""
    try:
        latest_tag = git(source, "describe", "--tags", "--abbrev=0", target)
    except subprocess.CalledProcessError:
        pass
    document = {
        "format": "curobo-metal-upstream-drift",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/NVlabs/curobo.git",
        "base_revision": base,
        "target_revision": target,
        "target_tag": latest_tag or None,
        "up_to_date": base == target,
        "summary": {"changed_files": len(entries), **buckets},
        "changes": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
