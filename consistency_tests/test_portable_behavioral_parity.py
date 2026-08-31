"""Release-blocking gates for credible portable behavioral parity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/gauntlet/portable-case-census.json"


def _report() -> dict:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["upstream_revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    return payload


def test_every_substituted_module_has_case_level_evidence() -> None:
    summary = _report()["summary"]
    assert summary["substituted_test_modules"] == 116
    assert summary["covered_test_modules"] == 116
    assert summary["missing_test_modules"] == []


def test_evidence_came_from_a_clean_installed_wheel_on_mps() -> None:
    execution = _report().get("execution", {})
    assert execution.get("installed_wheel") is True
    assert execution.get("candidate_wheel_record_verified") is True
    assert execution.get("verified_record_files", 0) >= 100
    assert execution.get("mps_available") is True
    assert execution.get("pytorch_enable_mps_fallback") == "0"
    assert "site-packages" in execution.get("curobo_import_path", "").split("/")


def test_every_pinned_source_and_portable_adaptation_is_hash_audited() -> None:
    source = _report().get("source", {})
    assert len(source.get("modules", {})) == 116
    adapter = source.get("portable_adapter", {})
    assert len(adapter.get("source_sha256", "")) == 64
    # One record per test module plus the adapted pinned conftest.
    assert len(adapter.get("records", {})) == 117
    assert adapter.get("device_string_replacements", 0) > 0
    assert adapter.get("availability_replacements", 0) > 0


def test_every_portable_case_executes_or_has_a_narrow_mechanism_exclusion() -> None:
    summary = _report()["summary"]
    assert summary["test_definition_complete"], summary["parity"]


def test_every_portable_case_matches_pinned_upstream_behavior() -> None:
    summary = _report()["summary"]
    assert summary["implementation_parity_complete"], summary["pytest"]
