from __future__ import annotations

from tools.gauntlet.measure_curobo_consistency import measure


def test_measurement_is_fail_closed_and_matches_the_red_suite() -> None:
    result = measure()
    metrics = result["metrics"]
    gaps = result["gaps"]
    assert metrics["runtime_modules_total"] == 361
    assert metrics["runtime_modules_present"] == 361
    assert metrics["upstream_workloads_total"] == 225
    assert metrics["portable_capabilities_total"] == 23
    assert metrics["runtime_modules_exact_static"] + gaps["static_runtime_modules"] == 361
    assert (
        metrics["upstream_workloads_reviewed"]
        + gaps["unreviewed_upstream_workloads"]
        == 225
    )
    assert (
        metrics["portable_capabilities_equivalent"]
        + gaps["capabilities_without_full_equivalence"]
        == 23
    )
    assert gaps[result["largest_gap"]] == max(gaps.values())
    assert 0.0 <= result["backlog_completion_score"] <= 1.0
