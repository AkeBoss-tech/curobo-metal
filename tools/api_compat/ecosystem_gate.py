#!/usr/bin/env python3
"""Resolve every real-world compatibility target against the local package."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    results = []
    modules: dict[str, object] = {}
    for item in corpus["symbols"]:
        if not item["compatibility_target"]:
            continue
        module_name = item["module"]
        try:
            module = modules.get(module_name)
            if module is None:
                module = importlib.import_module(module_name)
                modules[module_name] = module
            if item["name"] is not None:
                try:
                    getattr(module, item["name"])
                except AttributeError:
                    # ``from package import child`` asks the import system to
                    # resolve ``package.child`` even when __init__ does not
                    # eagerly bind it.
                    importlib.import_module(f"{module_name}.{item['name']}")
            status, error = "resolved", None
        except Exception as exc:  # report the observable import failure verbatim
            status = "unresolved"
            error = f"{type(exc).__name__}: {exc}"
        results.append({
            "identifier": item["identifier"],
            "status": status,
            "error": error,
            "classifications": item["classifications"],
        })
    unresolved = [item for item in results if item["status"] != "resolved"]
    report = {
        "schema_version": 1,
        "corpus_revision": corpus["upstream"]["revision"],
        "summary": {
            "targets": len(results),
            "resolved": len(results) - len(unresolved),
            "unresolved": len(unresolved),
        },
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
