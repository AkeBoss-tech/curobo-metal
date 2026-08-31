#!/usr/bin/env python3
"""Promote a fully passing clean-wheel batch into the pinned workload census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.api_compat.upstream_execution_census import PINNED_REVISION, _summary, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("artifacts/api_compat/upstream-api.json"))
    parser.add_argument("--census", type=Path, default=Path("artifacts/api_compat/upstream-execution-census.json"))
    args = parser.parse_args(argv)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if artifact.get("upstream_revision") != PINNED_REVISION:
        parser.error("batch artifact has the wrong upstream revision")
    pytest = artifact.get("pytest", {})
    if (
        pytest.get("exit_code") != 0
        or pytest.get("failures") != 0
        or pytest.get("errors") != 0
        or pytest.get("skipped") != 0
        or pytest.get("passed") != pytest.get("tests")
    ):
        parser.error("batch artifact is not an all-pass, zero-skip execution")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    census = json.loads(args.census.read_text(encoding="utf-8"))
    validate(inventory, census)
    entries = {entry["module"]: entry for entry in census["entries"]}
    artifact_path = args.artifact.as_posix()
    for module, digest in artifact["modules"].items():
        entry = entries.get(module)
        if entry is None:
            parser.error(f"artifact module is absent from census: {module}")
        entry.update(
            disposition="applicable_unchanged",
            rationale="Pinned module passed unchanged against a clean installed curobo-metal wheel.",
            evidence=[
                f"{artifact_path} records zero-failure, zero-error, zero-skip clean-wheel execution.",
                f"Pinned source SHA-256: {digest}.",
            ],
        )
    census["summary"] = _summary(census["entries"])
    validate(inventory, census)
    args.census.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(census["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
