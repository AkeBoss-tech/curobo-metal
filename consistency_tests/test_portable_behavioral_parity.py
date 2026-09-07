"""Release-blocking gates for credible portable behavioral parity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/gauntlet/portable-case-census.json"
EXECUTION_CENSUS = ROOT / "artifacts/api_compat/upstream-execution-census.json"


def _report() -> dict:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["upstream_revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    return payload


def _expected_modules() -> set[str]:
    census = json.loads(EXECUTION_CENSUS.read_text(encoding="utf-8"))
    assert census["upstream_revision"] == _report()["upstream_revision"]
    modules = {
        entry["module"]
        for entry in census["entries"]
        if entry["surface"] == "bundled_test"
        and entry["disposition"] == "platform_substituted"
    }
    assert modules
    return modules


def test_every_substituted_module_has_case_level_evidence() -> None:
    report = _report()
    expected = _expected_modules()
    summary = report["summary"]
    assert {case["module"] for case in report["cases"]} == expected
    assert summary["substituted_test_modules"] == len(expected)
    assert summary["covered_test_modules"] == len(expected)
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
    expected = _expected_modules()
    assert set(source.get("modules", {})) == expected
    adapter = source.get("portable_adapter", {})
    assert len(adapter.get("source_sha256", "")) == 64
    # One record per test module plus the adapted pinned conftest.
    assert set(adapter.get("records", {})) == expected | {"conftest.py"}
    assert adapter.get("device_string_replacements", 0) > 0
    assert adapter.get("availability_replacements", 0) > 0


def test_every_portable_case_executes_or_has_a_narrow_mechanism_exclusion() -> None:
    summary = _report()["summary"]
    assert summary["test_definition_complete"], summary["parity"]


def test_every_portable_case_matches_pinned_upstream_behavior() -> None:
    summary = _report()["summary"]
    assert summary["implementation_parity_complete"], summary["pytest"]
