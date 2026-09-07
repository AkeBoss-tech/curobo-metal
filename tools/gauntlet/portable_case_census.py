#!/usr/bin/env python3
"""Build a fail-closed, case-level portable-parity census from pytest JUnit.

Unlike the legacy module census, a module containing the word ``cuda`` is not
considered reviewed. Every collected case must execute, or match a narrow
case-level mechanism exclusion in the policy. Collection errors and CUDA-gated
skips remain visible blockers.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_for(testcase: ET.Element, modules: tuple[str, ...]) -> str:
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    combined = f"{classname}.{name}".replace("/", ".")
    for module in modules:
        staged_alias = module.removeprefix("curobo.tests.")
        if module in combined or staged_alias in combined:
            return module
    raise ValueError(
        "JUnit testcase does not belong to the substituted portable surface: "
        f"classname={classname!r}, name={name!r}"
    )


def _case_id(testcase: ET.Element, module: str) -> str:
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    if not classname:
        return f"{module}::<collection>"
    suffix = classname.split(module, 1)[-1].lstrip(".")
    return "::".join(part for part in (module, suffix, name) if part)


def _pytest_status(testcase: ET.Element) -> tuple[str, str | None]:
    for tag, status in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        node = testcase.find(tag)
        if node is not None:
            return status, node.attrib.get("message") or (node.text or "").splitlines()[0]
    return "passed", None


def build_census(
    *,
    policy: dict[str, Any],
    execution_census: dict[str, Any],
    junit_path: Path,
) -> dict[str, Any]:
    revision = policy["upstream_revision"]
    if execution_census.get("upstream_revision") != revision:
        raise ValueError("execution census revision differs from portable policy")
    disposition = policy["source_disposition"]
    modules = tuple(
        sorted(
            [
                entry["module"]
                for entry in execution_census["entries"]
                if entry["surface"] == "bundled_test"
                and entry["disposition"] == disposition
            ],
            key=len,
            reverse=True,
        )
    )
    if not modules:
        raise ValueError(f"no bundled tests have disposition {disposition!r}")

    exclusions = policy.get("mechanism_exclusions", [])
    for record in exclusions:
        pattern = record.get("case_pattern", "")
        reason = record.get("reason", "")
        if not pattern or not reason or "*" == pattern:
            raise ValueError("mechanism exclusions require a narrow case_pattern and reason")

    root = ET.parse(junit_path).getroot()
    testsuite = root if root.tag == "testsuite" else root.find("testsuite")
    if testsuite is None:
        raise ValueError("JUnit document has no testsuite")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    covered_modules: set[str] = set()
    for testcase in testsuite.iter("testcase"):
        module = _module_for(testcase, modules)
        case_id = _case_id(testcase, module)
        if case_id in seen:
            raise ValueError(f"duplicate JUnit case ID: {case_id}")
        seen.add(case_id)
        covered_modules.add(module)
        pytest_status, detail = _pytest_status(testcase)
        matched = []
        for record in exclusions:
            pattern = record["case_pattern"]
            is_match = (
                case_id == pattern
                if "*" not in pattern and "?" not in pattern
                else fnmatch.fnmatchcase(case_id, pattern)
            )
            if is_match:
                matched.append(record)
        if len(matched) > 1:
            raise ValueError(f"case matches multiple mechanism exclusions: {case_id}")
        if matched:
            parity_status = "mechanism_only_excluded"
            exclusion = matched[0]
        elif pytest_status in {"passed", "failed"}:
            parity_status = "portable_executed"
            exclusion = None
        elif pytest_status == "error":
            parity_status = "portable_blocked"
            exclusion = None
        else:
            parity_status = "unreviewed_skip"
            exclusion = None
        cases.append(
            {
                "id": case_id,
                "module": module,
                "pytest_status": pytest_status,
                "parity_status": parity_status,
                "detail": detail,
                "exclusion": exclusion,
            }
        )

    missing_modules = sorted(set(modules) - covered_modules)
    status_counts: dict[str, int] = {}
    pytest_counts: dict[str, int] = {}
    area_counts: dict[str, dict[str, Any]] = {}
    portable_passed = 0
    portable_failed = 0
    excluded_raw_failures = 0
    for case in cases:
        status_counts[case["parity_status"]] = status_counts.get(case["parity_status"], 0) + 1
        pytest_counts[case["pytest_status"]] = pytest_counts.get(case["pytest_status"], 0) + 1
        suffix = case["module"].removeprefix("curobo.tests.")
        area = suffix.removeprefix("_src.").split(".", 1)[0]
        area_record = area_counts.setdefault(
            area,
            {
                "cases": 0,
                "pytest": {},
                "parity": {},
                "portable_passed": 0,
                "portable_failed": 0,
                "excluded_raw_failures": 0,
            },
        )
        area_record["cases"] += 1
        area_record["pytest"][case["pytest_status"]] = (
            area_record["pytest"].get(case["pytest_status"], 0) + 1
        )
        area_record["parity"][case["parity_status"]] = (
            area_record["parity"].get(case["parity_status"], 0) + 1
        )
        if case["parity_status"] == "portable_executed":
            if case["pytest_status"] == "passed":
                portable_passed += 1
                area_record["portable_passed"] += 1
            elif case["pytest_status"] == "failed":
                portable_failed += 1
                area_record["portable_failed"] += 1
        elif (
            case["parity_status"] == "mechanism_only_excluded"
            and case["pytest_status"] == "failed"
        ):
            excluded_raw_failures += 1
            area_record["excluded_raw_failures"] += 1
    definition_complete = (
        not missing_modules
        and status_counts.get("portable_blocked", 0) == 0
        and status_counts.get("unreviewed_skip", 0) == 0
    )
    implementation_complete = definition_complete and portable_failed == 0
    return {
        "schema_version": 1,
        "policy": policy["name"],
        "upstream_revision": revision,
        "junit": {"path": str(junit_path), "sha256": _sha256(junit_path)},
        "summary": {
            "substituted_test_modules": len(modules),
            "covered_test_modules": len(covered_modules),
            "missing_test_modules": missing_modules,
            "cases": len(cases),
            "pytest": pytest_counts,
            "parity": status_counts,
            "portable_passed": portable_passed,
            "portable_failed": portable_failed,
            "excluded_raw_failures": excluded_raw_failures,
            "areas": dict(sorted(area_counts.items())),
            "test_definition_complete": definition_complete,
            "implementation_parity_complete": implementation_complete,
        },
        "cases": sorted(cases, key=lambda item: item["id"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "gauntlet/portable-dropin-parity.json")
    parser.add_argument(
        "--execution-census",
        type=Path,
        default=ROOT / "artifacts/api_compat/upstream-execution-census.json",
    )
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provenance-from",
        type=Path,
        help="Prior runner report whose installed-wheel execution/source provenance is retained.",
    )
    args = parser.parse_args(argv)
    result = build_census(
        policy=_load(args.policy),
        execution_census=_load(args.execution_census),
        junit_path=args.junit.resolve(),
    )
    if args.provenance_from:
        provenance = _load(args.provenance_from)
        for key in ("execution", "source", "pytest_exit_code"):
            if key not in provenance:
                parser.error(f"provenance report is missing {key!r}")
            result[key] = provenance[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["test_definition_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
