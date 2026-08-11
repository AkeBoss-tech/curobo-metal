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
            "unreviewed": 225,
            "applicable_unchanged": 0,
            "platform_substituted": 0,
            "external_unavailable": 0,
            "not_applicable": 0,
        },
        "review_complete": False,
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
    changed["entries"][0]["disposition"] = "not_applicable"
    with pytest.raises(ValueError, match="reviewed entry has no evidence"):
        validate(INVENTORY, changed)


def test_release_gate_rejects_unreviewed_entries() -> None:
    with pytest.raises(ValueError, match="225 unreviewed"):
        validate(INVENTORY, CENSUS, require_reviewed=True)
