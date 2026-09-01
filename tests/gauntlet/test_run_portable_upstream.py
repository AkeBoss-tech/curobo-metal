from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from tools.gauntlet.run_portable_upstream import (
    PINNED_REVISION,
    ModuleRun,
    _raw_mechanism_scopes,
    _run_module,
    merge_module_junit,
    selected_test_paths,
)


def test_selects_every_substituted_bundled_test_as_a_safe_relative_path() -> None:
    census = {
        "entries": [
            {
                "module": "curobo.tests._src.types.test_pose",
                "surface": "bundled_test",
                "disposition": "platform_substituted",
            },
            {
                "module": "curobo.tests._src.types.test_device_cfg",
                "surface": "bundled_test",
                "disposition": "unchanged_upstream",
            },
        ]
    }
    assert PINNED_REVISION == "8e734f3ced1df898990bcd92de40abce475907db"
    assert selected_test_paths(census, "platform_substituted") == (
        "_src/types/test_pose.py",
    )


def test_module_watchdog_terminates_native_style_wedge(tmp_path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_wedge.py").write_text(
        "import time\ndef test_wedge():\n    time.sleep(60)\n", encoding="utf-8"
    )
    run = _run_module(
        module="curobo.tests.test_wedge",
        test_path="tests/test_wedge.py",
        junit_path=tmp_path / "wedge.xml",
        cwd=tmp_path,
        environment=os.environ.copy(),
        timeout_seconds=0.2,
        terminate_grace_seconds=0.1,
    )
    assert run.timed_out
    assert run.returncode == 124
    assert run.elapsed_seconds < 5


def test_raw_mechanism_scopes_are_narrow_and_policy_reviewed() -> None:
    module = "curobo.tests._src.optim.test_optimizer_cuda_graph"
    policy = {
        "mechanism_exclusions": [
            {
                "case_pattern": (
                    f"{module}::tests._src.optim.test_optimizer_cuda_graph."
                    "TestOptimizerCudaGraph::test_mixin_can_optimize"
                ),
                "reason": "Requires CUDA Graph capture.",
            },
            {
                "case_pattern": f"{module}::tests.x::test_float64_adapter_artifact",
                "reason": "MPS has no float64 storage.",
            },
        ]
    }
    assert _raw_mechanism_scopes(policy, module) == frozenset(
        {
            "TestOptimizerCudaGraph",
            "TestOptimizerCudaGraph.test_mixin_can_optimize",
        }
    )


def test_merge_retains_complete_shard_and_synthetic_timeout(tmp_path) -> None:
    complete = tmp_path / "complete.xml"
    complete.write_text(
        '<?xml version="1.0"?><testsuites><testsuite>'
        '<testcase classname="curobo.tests.test_ok" name="test_ok" time="0.1" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    runs = (
        ModuleRun(
            module="curobo.tests.test_ok",
            test_path="tests/test_ok.py",
            junit_path=complete,
            returncode=0,
            timed_out=False,
            elapsed_seconds=0.1,
            junit_complete=True,
        ),
        ModuleRun(
            module="curobo.tests.test_wedge",
            test_path="tests/test_wedge.py",
            junit_path=tmp_path / "missing.xml",
            returncode=124,
            timed_out=True,
            elapsed_seconds=0.2,
            junit_complete=False,
        ),
    )
    merged = tmp_path / "merged.xml"
    merge_module_junit(runs, merged)
    root = ET.parse(merged).getroot()
    cases = list(root.iter("testcase"))
    assert [(case.attrib["classname"], case.attrib["name"]) for case in cases] == [
        ("curobo.tests.test_ok", "test_ok"),
        ("curobo.tests.test_wedge", "<module-timeout>"),
    ]
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib["tests"] == "2"
    assert suite.attrib["errors"] == "1"
