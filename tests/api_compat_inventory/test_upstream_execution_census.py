from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.api_compat.upstream_execution_census import initialize, validate


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = json.loads((ROOT / "artifacts/api_compat/upstream-api.json").read_text())
CENSUS = json.loads((ROOT / "artifacts/api_compat/upstream-execution-census.json").read_text())


def test_checked_in_census_is_exact_and_fail_closed() -> None:
    validate(INVENTORY, CENSUS)
    assert CENSUS["summary"] == {
        "total": 225,
        "by_surface": {"bundled_test": 211, "bundled_example": 14},
        "by_disposition": {
            "unreviewed": 0,
            "applicable_unchanged": 55,
            "platform_substituted": 120,
            "external_unavailable": 15,
            "not_applicable": 35,
        },
        "review_complete": True,
    }


def test_initializer_is_deterministic() -> None:
    assert initialize(INVENTORY) == initialize(INVENTORY)


def test_missing_entry_is_rejected() -> None:
    changed = copy.deepcopy(CENSUS)
    changed["entries"].pop()
    with pytest.raises(ValueError, match="does not match pinned inventory"):
        validate(INVENTORY, changed)


def test_reviewed_entry_requires_evidence() -> None:
    changed = copy.deepcopy(CENSUS)
    entry = next(
        item for item in changed["entries"] if item["disposition"] == "applicable_unchanged"
    )
    entry["evidence"] = []
    with pytest.raises(ValueError, match="reviewed entry has no evidence"):
        validate(INVENTORY, changed)


def test_release_gate_accepts_complete_review() -> None:
    validate(INVENTORY, CENSUS, require_reviewed=True)


def test_source_classifications_are_backed_by_pinned_dependency_report() -> None:
    report = json.loads(
        (ROOT / "artifacts/gauntlet/upstream-workload-dependencies.json").read_text()
    )
    assert report["upstream_revision"] == CENSUS["upstream_revision"]
    records = {entry["module"]: entry for entry in report["entries"]}
    automatically_reviewed = [
        entry
        for entry in CENSUS["entries"]
        if any("upstream-workload-dependencies.json" in item for item in entry["evidence"])
    ]
    assert len(automatically_reviewed) == 156
    for entry in automatically_reviewed:
        record = records[entry["module"]]
        assert record["sha256"] in entry["evidence"][0]


def test_bundled_examples_are_reviewed_without_unexecuted_pass_claims() -> None:
    examples = [entry for entry in CENSUS["entries"] if entry["surface"] == "bundled_example"]
    assert len(examples) == 14
    assert all(entry["disposition"] != "unreviewed" for entry in examples)
    assert all(entry["disposition"] != "applicable_unchanged" for entry in examples)
    assert all(entry["evidence"] for entry in examples)
