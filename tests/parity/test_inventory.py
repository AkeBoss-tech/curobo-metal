import json
from pathlib import Path


def test_committed_inventory_is_complete_and_pinned():
    data = json.loads(Path("artifacts/parity/capabilities.json").read_text())
    assert data["generated_from"]["revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    records = data["capabilities"]
    assert len(records) >= 15
    assert len({item["id"] for item in records}) == len(records)
    assert sum(data["summary"].values()) == len(records)
    assert all(item["blocker"] for item in records)
    assert all(item["upstream_evidence"]["path"] for item in records)


def test_no_full_port_classification_exists():
    data = json.loads(Path("artifacts/parity/capabilities.json").read_text())
    allowed = {"compatible", "semantically_equivalent", "partial", "unsupported", "not_applicable"}
    assert {item["classification"] for item in data["capabilities"]} <= allowed
    assert any(item["classification"] == "unsupported" for item in data["capabilities"])
    assert any(item["classification"] == "partial" for item in data["capabilities"])
