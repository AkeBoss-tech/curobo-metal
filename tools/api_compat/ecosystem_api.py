#!/usr/bin/env python3
"""Build the real-world cuRobo import corpus used by the stable release gate."""

from __future__ import annotations

import argparse
import ast
import base64
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
IMPORT_LINE = re.compile(
    r"^\s*(?:from\s+(curobo(?:\.[\w.]+)?)\s+import\s+([\w., ()]+)|import\s+(curobo(?:\.[\w.]+)?)(?:\s|$))",
    re.MULTILINE,
)


def _imports_from_python(text: str, source: str) -> list[tuple[str, str, str]]:
    try:
        tree = ast.parse(text, filename=source)
    except SyntaxError:
        return _imports_from_text(text, source)
    found: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom) and node.module
            and (node.module == "curobo" or node.module.startswith("curobo."))
        ):
            for alias in node.names:
                if alias.name != "*":
                    found.append((node.module, alias.name, source))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "curobo" or alias.name.startswith("curobo."):
                    found.append((alias.name, "", source))
    return found


def _imports_from_text(text: str, source: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for match in IMPORT_LINE.finditer(text):
        module = match.group(1) or match.group(3)
        names = match.group(2)
        if not names:
            found.append((module, "", source))
            continue
        for name in names.replace("(", "").replace(")", "").split(","):
            name = name.strip().split()[0] if name.strip() else ""
            if name:
                found.append((module, name, source))
    return found


def _scan_files(paths: Iterable[Path], root: Path) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        source = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            found.extend(_imports_from_python(text, source))
        else:
            found.extend(_imports_from_text(text, source))
    return found


def _gh_json(*arguments: str) -> object:
    result = subprocess.run(
        ["gh", "api", *arguments], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def _downstream_imports(limit: int) -> tuple[list[tuple[str, str, str]], list[dict]]:
    result = _gh_json(
        "search/code", "-X", "GET", "-f",
        'q="from curobo" language:Python -repo:NVlabs/curobo',
        "-f", f"per_page={min(limit, 100)}",
    )
    imports: list[tuple[str, str, str]] = []
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in result.get("items", [])[:limit]:
        repository = item["repository"]["full_name"]
        path = item["path"]
        if (repository, path) in seen:
            continue
        seen.add((repository, path))
        # Ignore vendored cuRobo source trees; they describe an implementation,
        # rather than a downstream application's compatibility needs.
        normalized = path.lstrip("./")
        if normalized.startswith(("curobo/", "src/curobo/")):
            continue
        payload = _gh_json(f"repos/{repository}/contents/{path}")
        if payload.get("encoding") != "base64" or "content" not in payload:
            continue
        text = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
        source = f"github:{repository}/{path}@{payload.get('sha', 'unknown')}"
        file_imports = _imports_from_python(text, source)
        if not file_imports:
            continue
        imports.extend(file_imports)
        sources.append({
            "repository": repository,
            "path": path,
            "blob_sha": payload.get("sha"),
            "url": item.get("html_url"),
        })
    return imports, sources


def _mechanism(module: str) -> str | None:
    parts = set(module.split("."))
    if parts & {"cuda", "curobolib", "warp", "cuda_graph_util"}:
        return "CUDA_MECHANISM"
    if parts & {"usd_helper", "isaac_sim", "ros", "feature_mapping"}:
        return "EXTERNAL_INTEGRATION"
    if "._src." in f".{module}.":
        return "INTERNAL"
    return None


def build(upstream: Path, downstream_limit: int, skip_github: bool) -> dict:
    upstream = upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        raise SystemExit(f"expected upstream {PINNED_REVISION}, found {revision}")

    groups = {
        "PUBLIC_DOCUMENTED": _scan_files(
            list((upstream / "docs").rglob("*.rst"))
            + list((upstream / "docs").rglob("*.md")), upstream,
        ),
        "PUBLIC_EXAMPLE_USED": _scan_files(
            (upstream / "curobo/examples").rglob("*.py"), upstream,
        ),
        "UPSTREAM_TEST_USED": _scan_files(
            (upstream / "curobo/tests").rglob("*.py"), upstream,
        ),
    }
    downstream_sources: list[dict] = []
    if not skip_github:
        groups["ECOSYSTEM_USED"], downstream_sources = _downstream_imports(downstream_limit)

    records: dict[str, dict] = defaultdict(
        lambda: {"classifications": set(), "sources": set()}
    )
    for classification, entries in groups.items():
        for module, name, source in entries:
            identifier = f"{module}.{name}" if name else module
            records[identifier]["module"] = module
            records[identifier]["name"] = name or None
            records[identifier]["classifications"].add(classification)
            records[identifier]["sources"].add(source)
            special = _mechanism(module)
            if special:
                records[identifier]["classifications"].add(special)

    target_classes = {"PUBLIC_DOCUMENTED", "PUBLIC_EXAMPLE_USED", "ECOSYSTEM_USED"}
    excluded_classes = {"CUDA_MECHANISM", "EXTERNAL_INTEGRATION"}
    symbols = []
    for identifier, record in sorted(records.items()):
        classifications = sorted(record["classifications"])
        target = bool(target_classes.intersection(classifications)) and not bool(
            excluded_classes.intersection(classifications)
        )
        symbols.append({
            "identifier": identifier,
            "module": record["module"],
            "name": record["name"],
            "classifications": classifications,
            "compatibility_target": target,
            "sources": sorted(record["sources"]),
        })
    return {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "upstream": {
            "repository": "https://github.com/NVlabs/curobo.git",
            "revision": revision,
        },
        "method": {
            "documented": "imports in pinned upstream docs",
            "example_used": "imports in pinned upstream examples",
            "test_used": "imports in pinned upstream tests",
            "ecosystem_used": "GitHub code search with vendored cuRobo trees excluded",
            "downstream_limit": downstream_limit,
        },
        "summary": {
            "symbols": len(symbols),
            "compatibility_targets": sum(item["compatibility_target"] for item in symbols),
            "downstream_files": len(downstream_sources),
        },
        "downstream_sources": downstream_sources,
        "symbols": symbols,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--downstream-limit", type=int, default=100)
    parser.add_argument("--skip-github", action="store_true")
    args = parser.parse_args()
    report = build(args.upstream, args.downstream_limit, args.skip_github)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
