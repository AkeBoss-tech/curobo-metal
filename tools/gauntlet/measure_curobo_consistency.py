#!/usr/bin/env python3
"""Measure the pinned cuRoboV2 consistency gauntlet from committed evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.api_compat.surface_gate import build_report
from tools.api_compat.upstream_execution_census import validate


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "gauntlet/curobo-consistency.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def measure(root: Path = ROOT) -> dict[str, Any]:
    config = _load(root / "gauntlet/curobo-consistency.json")
    upstream = _load(root / "artifacts/api_compat/upstream-api.json")
    census = _load(root / "artifacts/api_compat/upstream-execution-census.json")
    capabilities = _load(root / "artifacts/parity/capabilities.json")

    revision = config["upstream_revision"]
    if upstream.get("upstream", {}).get("revision") != revision:
        raise ValueError("API inventory revision differs from the gauntlet")
    if census.get("upstream_revision") != revision:
        raise ValueError("execution census revision differs from the gauntlet")
    if capabilities.get("generated_from", {}).get("revision") != revision:
        raise ValueError("capability inventory revision differs from the gauntlet")
    validate(upstream, census)

    surface = build_report(upstream, root / "src")
    rows = surface["modules"]
    exact_modules = sum(
        row["module"] == "present"
        and not row["exports"]["missing"]
        and not row["callables"]["different"]
        for row in rows
    )
    reviewed = sum(
        entry["disposition"] != "unreviewed" for entry in census["entries"]
    )
    portable = [
        record
        for record in capabilities["capabilities"]
        if record["classification"]
        not in {"intentionally_platform_inapplicable", "external_integration_only"}
    ]
    equivalent = sum(
        record["classification"] == "semantically_equivalent" for record in portable
    )

    metrics = {
        "runtime_modules_total": len(rows),
        "runtime_modules_present": surface["summary"]["present_modules"],
        "runtime_modules_exact_static": exact_modules,
        "missing_exports": surface["summary"]["expected_exports"]
        - surface["summary"]["present_exports"],
        "different_callable_shapes": surface["summary"]["different_callable_shapes"],
        "upstream_workloads_total": len(census["entries"]),
        "upstream_workloads_reviewed": reviewed,
        "upstream_workloads_unreviewed": len(census["entries"]) - reviewed,
        "portable_capabilities_total": len(portable),
        "portable_capabilities_equivalent": equivalent,
    }
    ratios = {
        "runtime_modules_present_ratio": _ratio(
            metrics["runtime_modules_present"], metrics["runtime_modules_total"]
        ),
        "runtime_modules_exact_static_ratio": _ratio(
            metrics["runtime_modules_exact_static"], metrics["runtime_modules_total"]
        ),
        "upstream_workloads_reviewed_ratio": _ratio(
            metrics["upstream_workloads_reviewed"], metrics["upstream_workloads_total"]
        ),
        "portable_capabilities_equivalent_ratio": _ratio(
            metrics["portable_capabilities_equivalent"],
            metrics["portable_capabilities_total"],
        ),
    }
    weights = config["diagnostic_weights"]
    score = sum(ratios[name] * weight for name, weight in weights.items())
    gaps = {
        "static_runtime_modules": metrics["runtime_modules_total"]
        - metrics["runtime_modules_exact_static"],
        "unreviewed_upstream_workloads": metrics["upstream_workloads_unreviewed"],
        "capabilities_without_full_equivalence": metrics["portable_capabilities_total"]
        - metrics["portable_capabilities_equivalent"],
    }
    largest_gap = max(gaps, key=gaps.get)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "schema_version": 1,
        "gauntlet": config["name"],
        "upstream_revision": revision,
        "local_revision": head,
        "metrics": metrics,
        "ratios": ratios,
        "backlog_completion_score": score,
        "gaps": gaps,
        "largest_gap": largest_gap,
        "backlog_targets_met": all(
            ratios[name] >= target
            for name, target in config["backlog_targets"].items()
            if name in ratios
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = measure(args.repository.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["backlog_targets_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
