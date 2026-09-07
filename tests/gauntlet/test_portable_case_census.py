from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.gauntlet.portable_case_census import build_census


REVISION = "8e734f3ced1df898990bcd92de40abce475907db"


def _inputs(tmp_path):
    policy = {
        "name": "test-policy",
        "upstream_revision": REVISION,
        "source_disposition": "platform_substituted",
        "mechanism_exclusions": [],
    }
    census = {
        "upstream_revision": REVISION,
        "entries": [
            {
                "module": "curobo.tests._src.types.test_pose",
                "surface": "bundled_test",
                "disposition": "platform_substituted",
            }
        ],
    }
    junit = tmp_path / "junit.xml"
    return policy, census, junit


def test_failures_are_executed_evidence_but_skips_are_not(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    suite = ET.Element("testsuite")
    ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_pass",
    )
    failed = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_failure",
    )
    ET.SubElement(failed, "failure", message="different value")
    skipped = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_cuda",
    )
    ET.SubElement(skipped, "skipped", message="CUDA required")
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)

    assert result["summary"]["pytest"] == {"passed": 1, "failed": 1, "skipped": 1}
    assert result["summary"]["parity"] == {
        "portable_executed": 2,
        "unreviewed_skip": 1,
    }
    assert result["summary"]["portable_passed"] == 1
    assert result["summary"]["portable_failed"] == 1
    assert result["summary"]["excluded_raw_failures"] == 0
    assert result["summary"]["areas"]["types"]["cases"] == 3
    assert result["summary"]["areas"]["types"]["portable_passed"] == 1
    assert result["summary"]["areas"]["types"]["portable_failed"] == 1
    assert result["summary"]["areas"]["types"]["excluded_raw_failures"] == 0
    assert not result["summary"]["test_definition_complete"]


def test_excluded_raw_failures_are_separate_from_portable_failures(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    policy["mechanism_exclusions"] = [
        {
            "case_pattern": "*::test_raw_cuda_graph",
            "reason": "Raw CUDA graph capture has no Metal object-level equivalent.",
        }
    ]
    suite = ET.Element("testsuite")
    ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_pass",
    )
    portable_failure = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_portable_failure",
    )
    ET.SubElement(portable_failure, "failure", message="portable mismatch")
    excluded_failure = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_raw_cuda_graph",
    )
    ET.SubElement(excluded_failure, "failure", message="CUDA graph unavailable")
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)
    summary = result["summary"]

    # The legacy raw pytest count remains two failures, while the additive
    # fields identify the one actionable portable failure precisely.
    assert summary["pytest"]["failed"] == 2
    assert summary["portable_passed"] == 1
    assert summary["portable_failed"] == 1
    assert summary["excluded_raw_failures"] == 1
    area = summary["areas"]["types"]
    assert area["portable_passed"] == 1
    assert area["portable_failed"] == 1
    assert area["excluded_raw_failures"] == 1
    assert not summary["implementation_parity_complete"]


def test_excluded_failure_does_not_block_implementation_parity(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    policy["mechanism_exclusions"] = [
        {
            "case_pattern": "*::test_raw_cuda_graph",
            "reason": "Raw CUDA graph capture has no Metal object-level equivalent.",
        }
    ]
    suite = ET.Element("testsuite")
    ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_pass",
    )
    excluded_failure = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_raw_cuda_graph",
    )
    ET.SubElement(excluded_failure, "failure", message="CUDA graph unavailable")
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)
    summary = result["summary"]

    assert summary["portable_failed"] == 0
    assert summary["excluded_raw_failures"] == 1
    assert summary["implementation_parity_complete"]


def test_narrow_case_exclusion_closes_only_its_matching_case(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    policy["mechanism_exclusions"] = [
        {
            "case_pattern": "*::test_raw_cuda_graph",
            "reason": "Raw CUDA graph capture has no Metal object-level equivalent.",
        }
    ]
    suite = ET.Element("testsuite")
    skipped = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_raw_cuda_graph",
    )
    ET.SubElement(skipped, "skipped", message="CUDA required")
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)

    assert result["summary"]["parity"] == {"mechanism_only_excluded": 1}
    assert result["summary"]["test_definition_complete"]


def test_staged_test_module_names_map_back_to_canonical_upstream_module(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    suite = ET.Element("testsuite")
    ET.SubElement(
        suite,
        "testcase",
        classname="_src.types.test_pose.TestPose",
        name="test_staged",
    )
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)

    assert result["cases"][0]["module"] == "curobo.tests._src.types.test_pose"


def test_exact_exclusion_treats_parameter_brackets_literally(tmp_path) -> None:
    policy, census, junit = _inputs(tmp_path)
    case_id = "curobo.tests._src.types.test_pose::TestPose::test_grad[mps]"
    policy["mechanism_exclusions"] = [
        {"case_pattern": case_id, "reason": "MPS cannot represent float64 gradcheck inputs."}
    ]
    suite = ET.Element("testsuite")
    failed = ET.SubElement(
        suite,
        "testcase",
        classname="curobo.tests._src.types.test_pose.TestPose",
        name="test_grad[mps]",
    )
    ET.SubElement(failed, "failure", message="float64 unavailable")
    ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)

    result = build_census(policy=policy, execution_census=census, junit_path=junit)

    assert result["cases"][0]["parity_status"] == "mechanism_only_excluded"


def test_real_policy_keeps_warp_exclusions_case_level_and_lifecycle_portable() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "gauntlet/portable-dropin-parity.json").read_text())
    patterns = {record["case_pattern"] for record in policy["mechanism_exclusions"]}

    expected_counts = {
        "perception.mapper.test_block_checkpoint": 1,
        "perception.mapper.test_block_hash": 16,
        "perception.test_wp_mesh_sdf_alignment": 13,
        "perception.test_mesh_robot.TestSurfacePointValidity": 3,
        "geom.sdf.test_voxel_collision": 33,
    }
    for needle, expected in expected_counts.items():
        assert sum(needle in pattern for pattern in patterns) == expected

    block_hash = {pattern for pattern in patterns if "test_block_hash" in pattern}
    for portable_case in ("test_reset", "test_memory_usage", "test_get_stats"):
        assert not any(pattern.endswith(f"::{portable_case}") for pattern in block_hash)
    assert not any("::<collection>" in pattern for pattern in patterns)
    assert not any("*" in pattern or "?" in pattern for pattern in patterns)
