"""Intentionally red gates for complete pinned cuRoboV2 consistency.

These tests describe the stable drop-in target rather than the bounded alpha
release contract.  Keep them outside the default ``tests/`` tree until they are
green; run them explicitly while closing compatibility work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.api_compat.surface_gate import build_report
from tools.api_compat.upstream_execution_census import validate


ROOT = Path(__file__).resolve().parents[1]
PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"


def _json(path: str) -> dict[str, Any]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


UPSTREAM_INVENTORY = _json("artifacts/api_compat/upstream-api.json")
EXECUTION_CENSUS = _json("artifacts/api_compat/upstream-execution-census.json")
CAPABILITY_INVENTORY = _json("artifacts/parity/capabilities.json")
SURFACE_REPORT = build_report(UPSTREAM_INVENTORY, ROOT / "src")


def _runtime_id(row: dict[str, Any]) -> str:
    return row["name"]


def _workload_id(entry: dict[str, Any]) -> str:
    return entry["module"]


def _capability_id(record: dict[str, Any]) -> str:
    return record["id"]


def test_consistency_inputs_are_pinned_and_complete() -> None:
    """Refuse to report progress against stale or incomplete inventories."""
    assert UPSTREAM_INVENTORY["upstream"]["revision"] == PINNED_REVISION
    assert EXECUTION_CENSUS["upstream_revision"] == PINNED_REVISION
    assert CAPABILITY_INVENTORY["generated_from"]["revision"] == PINNED_REVISION
    assert SURFACE_REPORT["summary"]["modules"] == 361
    assert EXECUTION_CENSUS["summary"]["total"] == 225
    validate(UPSTREAM_INVENTORY, EXECUTION_CENSUS)


@pytest.mark.parametrize("row", SURFACE_REPORT["modules"], ids=_runtime_id)
def test_every_runtime_module_matches_static_upstream_surface(row: dict[str, Any]) -> None:
    """Require every module's exports and declared callables to match upstream."""
    problems: list[str] = []
    if row["module"] != "present":
        problems.append("module is missing")
    missing = row["exports"]["missing"]
    if missing:
        problems.append(f"missing exports ({len(missing)}): {missing}")
    different = row["callables"]["different"]
    if different:
        problems.append(f"different callable shapes ({len(different)}): {different}")
    assert not problems, "; ".join(problems)


@pytest.mark.parametrize("entry", EXECUTION_CENSUS["entries"], ids=_workload_id)
def test_every_upstream_test_or_example_has_been_reviewed(entry: dict[str, Any]) -> None:
    """Make each unclassified upstream workload a separately actionable failure."""
    assert entry["disposition"] != "unreviewed", (
        f"{entry['module']} has not been classified for unchanged execution, "
        "platform substitution, external unavailability, or non-applicability"
    )
    assert entry["evidence"], f"{entry['module']} has no review or execution evidence"


PORTABLE_CAPABILITIES = [
    record
    for record in CAPABILITY_INVENTORY["capabilities"]
    if record["classification"]
    not in {"intentionally_platform_inapplicable", "external_integration_only"}
]


@pytest.mark.parametrize("record", PORTABLE_CAPABILITIES, ids=_capability_id)
def test_every_portable_capability_has_full_upstream_semantic_evidence(
    record: dict[str, Any],
) -> None:
    """Require broad upstream evidence, not merely a local implementation."""
    assert record["classification"] == "semantically_equivalent", (
        f"{record['id']} is {record['classification']}: "
        f"{record.get('evidence_needed') or record['boundary']}"
    )
    assert record["test_evidence"], f"{record['id']} has no executable test evidence"
    assert not record["implementable_gaps"], (
        f"{record['id']} has implementation gaps: {record['implementable_gaps']}"
    )
