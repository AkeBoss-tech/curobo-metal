from __future__ import annotations

import xml.etree.ElementTree as ET

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
    assert result["summary"]["areas"]["types"]["cases"] == 3
    assert not result["summary"]["test_definition_complete"]


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
