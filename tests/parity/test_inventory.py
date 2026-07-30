import json
from pathlib import Path


def test_committed_inventory_is_complete_and_pinned():
    data = json.loads(Path("artifacts/parity/capabilities.json").read_text())
    assert data["schema_version"] == 2
    assert data["generated_from"]["revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    records = data["capabilities"]
    assert len(records) >= 25
    assert len({item["id"] for item in records}) == len(records)
    assert sum(data["summary"].values()) == len(records)
    assert all(item["boundary"] for item in records)
    assert all(item["upstream_evidence"]["path"] for item in records)
    assert all(
        item["local_evidence"] is not None
        for item in records
        if item["classification"] not in {
            "intentionally_platform_inapplicable",
            "external_integration_only",
        }
    )
    assert all(Path(path).is_file() for item in records for path in item["test_evidence"])


def test_no_full_port_classification_exists():
    data = json.loads(Path("artifacts/parity/capabilities.json").read_text())
    allowed = {
        "semantically_equivalent",
        "partial",
        "intentionally_platform_inapplicable",
        "external_integration_only",
        "evidence_blocked",
    }
    classifications = {item["classification"] for item in data["capabilities"]}
    assert classifications <= allowed
    assert "partial" not in classifications
    assert set(data["classification_vocabulary"]) == allowed
    assert data["summary"]["partial"] == 0
    assert data["summary"]["evidence_blocked"] > 0


def test_equivalence_and_blocked_claims_carry_the_required_evidence():
    records = json.loads(Path("artifacts/parity/capabilities.json").read_text())["capabilities"]
    equivalent = [item for item in records if item["classification"] == "semantically_equivalent"]
    assert equivalent
    assert all(item["test_evidence"] and not item["implementable_gaps"] for item in equivalent)

    blocked = [item for item in records if item["classification"] == "evidence_blocked"]
    assert blocked
    assert all(item["evidence_needed"] for item in blocked)
    # Evidence requirements are not portable implementation gaps.
    assert all(not item["implementable_gaps"] for item in blocked)


def test_every_partial_record_names_an_implementable_gap():
    records = json.loads(Path("artifacts/parity/capabilities.json").read_text())["capabilities"]
    assert all(
        item["implementable_gaps"]
        for item in records
        if item["classification"] == "partial"
    )
