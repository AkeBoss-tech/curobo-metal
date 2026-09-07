"""The compatibility gate must reject tampering and mismatched evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import pytest

from tools.application_compat.run import compare, compare_value, digest, verify_suite

SUITE = Path(__file__).resolve().parents[2] / "examples/application_compat"


def test_source_tampering_is_rejected(tmp_path):
    suite = tmp_path / "suite"
    shutil.copytree(SUITE, suite)
    verify_suite(suite)
    with (suite / "03_fk_franka.py").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_suite(suite)


def test_unhashed_python_file_is_rejected(tmp_path):
    suite = tmp_path / "suite"
    shutil.copytree(SUITE, suite)
    (suite / "curobo.py").write_text("# must not shadow the installed namespace\n")
    with pytest.raises(ValueError, match="inventory"):
        verify_suite(suite)


def test_comparison_checks_values_schema_and_status():
    base = {"shape": [1], "dtype": "torch.float32", "device": "mps", "values": [0.5]}
    cuda = {**base, "device": "cuda", "values": [0.500001]}
    assert not compare_value(base, cuda, atol=1e-4, rtol=0)
    assert compare_value(base, {**cuda, "values": [0.7]}, atol=1e-4, rtol=0)
    assert compare_value(base, {**cuda, "dtype": "torch.float64"}, atol=1, rtol=1)
    assert compare_value(base, {**cuda, "shape": [2]}, atol=1, rtol=1)
    assert compare_value(False, True, atol=1, rtol=1)
    assert compare_value(float("nan"), float("nan"), atol=1, rtol=1)


def _report(backend, device):
    manifest = verify_suite(SUITE)
    return {"backend": backend, "expected_device": device, "passed": True,
            "manifest_sha256": digest(SUITE / "manifest.json"), "source_hashes": manifest["files"],
            "upstream_revision": manifest["upstream_revision"],
            "cases": [{"file": c["file"], "passed": True,
                       "observations": {"value": {"shape": [1], "dtype": "torch.float32", "device": device, "values": [1.0]}}}
                      for c in manifest["cases"]]}


@pytest.mark.parametrize("mutation", ["missing_case", "source_hash", "residency", "failed_case"])
def test_comparison_rejects_invalid_reports(tmp_path, mutation):
    metal, cuda = _report("metal", "mps"), _report("cuda", "cuda")
    if mutation == "missing_case":
        cuda["cases"].pop()
    elif mutation == "source_hash":
        cuda["source_hashes"] = {}
    elif mutation == "residency":
        cuda["cases"][0]["observations"]["value"]["device"] = "cpu"
    else:
        cuda["cases"][0]["passed"] = False
    for name, value in (("metal", metal), ("cuda", cuda)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    args = SimpleNamespace(suite=SUITE, metal=tmp_path / "metal.json", cuda=tmp_path / "cuda.json", output=tmp_path / "comparison.json")
    assert compare(args) == 1
    assert not json.loads(args.output.read_text())["passed"]
