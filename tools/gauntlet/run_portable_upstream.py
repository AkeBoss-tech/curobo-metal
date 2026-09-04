#!/usr/bin/env python3
"""Replay every substituted pinned-upstream test against an installed wheel."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.gauntlet.portable_case_census import build_census
from tools.gauntlet.portable_test_adapter import adapt_source


ROOT = Path(__file__).resolve().parents[2]
PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
PYTEST_CASE_TIMEOUT_SECONDS = 300.0
DEFAULT_MODULE_TIMEOUT_SECONDS = 330.0
DEFAULT_TERMINATE_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class ModuleRun:
    module: str
    test_path: str
    junit_path: Path
    returncode: int
    timed_out: bool
    elapsed_seconds: float
    junit_complete: bool


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_installed_record(distribution_root: Path, wheel: Path) -> int:
    """Prove installed files match every SHA-256 entry in the candidate wheel."""

    with zipfile.ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError(f"candidate wheel has {len(records)} RECORD files")
        rows = list(csv.reader(archive.read(records[0]).decode("utf-8").splitlines()))
    checked = 0
    for relative, encoded_hash, _size in rows:
        if not encoded_hash:
            continue
        algorithm, separator, expected = encoded_hash.partition("=")
        if separator != "=" or algorithm != "sha256":
            raise ValueError(f"unsupported wheel RECORD hash for {relative}: {encoded_hash}")
        installed = distribution_root / relative
        if not installed.is_file():
            raise ValueError(f"installed candidate is missing wheel file: {relative}")
        digest = hashlib.sha256(installed.read_bytes()).digest()
        actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        if actual != expected:
            raise ValueError(f"installed file differs from candidate wheel: {relative}")
        checked += 1
    if checked < 100:
        raise ValueError(f"candidate wheel RECORD verified only {checked} files")
    return checked


def selected_test_paths(execution_census: dict[str, Any], disposition: str) -> tuple[str, ...]:
    paths = []
    for entry in execution_census["entries"]:
        if entry["surface"] != "bundled_test" or entry["disposition"] != disposition:
            continue
        module = entry["module"]
        prefix = "curobo.tests."
        if not module.startswith(prefix):
            raise ValueError(f"unexpected bundled-test module: {module}")
        paths.append(module.removeprefix(prefix).replace(".", "/") + ".py")
    if len(paths) != len(set(paths)):
        raise ValueError("execution census contains duplicate portable test paths")
    return tuple(sorted(paths))


def _module_for_path(relative: str) -> str:
    return f"curobo.tests.{relative.removesuffix('.py').replace('/', '.')}"


def _raw_mechanism_scopes(policy: dict[str, Any], module: str) -> frozenset[str]:
    """Return policy-reviewed test scopes whose real CUDA/Warp gate must remain."""

    scopes = set()
    mechanism_words = ("cuda graph", "cuda stream", "cuda event", "warp")
    for record in policy.get("mechanism_exclusions", []):
        pattern = record.get("case_pattern", "")
        reason = record.get("reason", "").lower()
        if not pattern.startswith(f"{module}::") or not any(
            word in reason for word in mechanism_words
        ):
            continue
        parts = pattern.split("::")
        test_name = parts[-1].split("[", 1)[0]
        classname = parts[-2].rsplit(".", 1)[-1] if len(parts) >= 3 else ""
        if classname.startswith("Test"):
            scopes.add(classname)
            scopes.add(f"{classname}.{test_name}")
        else:
            scopes.add(test_name)
    return frozenset(scopes)


def _terminate_worker(process: subprocess.Popen[bytes], grace_seconds: float) -> int:
    """Terminate a pytest process group, escalating when native code ignores SIGTERM."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def _run_module(
    *,
    module: str,
    test_path: str,
    junit_path: Path,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    terminate_grace_seconds: float,
) -> ModuleRun:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "--import-mode=importlib",
        "--continue-on-collection-errors",
        f"--junitxml={junit_path}",
        test_path,
    ]
    # The outer process-group watchdog is authoritative and works in the
    # clean wheel environment even when the optional pytest-timeout plugin is
    # absent. Add the finer per-case guard only when that plugin is installed.
    if importlib.util.find_spec("pytest_timeout") is not None:
        command[command.index(f"--junitxml={junit_path}"):command.index(f"--junitxml={junit_path}")] = [
            f"--timeout={PYTEST_CASE_TIMEOUT_SECONDS:g}",
            "--timeout-method=signal",
        ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_worker(process, terminate_grace_seconds)
        returncode = 124
    elapsed = time.monotonic() - started
    junit_complete = False
    if junit_path.is_file():
        try:
            ET.parse(junit_path)
            junit_complete = True
        except ET.ParseError:
            pass
    return ModuleRun(
        module=module,
        test_path=test_path,
        junit_path=junit_path,
        returncode=returncode,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
        junit_complete=junit_complete,
    )


def _synthetic_case(run: ModuleRun) -> ET.Element:
    if run.timed_out:
        name = "<module-timeout>"
        message = (
            f"module exceeded outer wall-clock watchdog of {run.elapsed_seconds:.3f}s; "
            "pytest worker was terminated"
        )
        error_type = "ModuleTimeout"
    else:
        name = "<module-crash>"
        message = f"pytest exited {run.returncode} without complete JUnit"
        error_type = "ModuleCrash"
    testcase = ET.Element(
        "testcase",
        classname=run.module,
        name=name,
        time=f"{run.elapsed_seconds:.6f}",
    )
    ET.SubElement(testcase, "error", message=message, type=error_type).text = message
    return testcase


def merge_module_junit(runs: tuple[ModuleRun, ...], destination: Path) -> None:
    """Merge every module shard and retain explicit crash/timeout census records."""

    suite = ET.Element("testsuite", name="portable-upstream-module-shards")
    for run in runs:
        if run.junit_complete:
            root = ET.parse(run.junit_path).getroot()
            for testcase in root.iter("testcase"):
                suite.append(copy.deepcopy(testcase))
        if run.timed_out or not run.junit_complete:
            suite.append(_synthetic_case(run))
    tests = list(suite.iter("testcase"))
    failures = sum(case.find("failure") is not None for case in tests)
    errors = sum(case.find("error") is not None for case in tests)
    skipped = sum(case.find("skipped") is not None for case in tests)
    elapsed = sum(float(case.attrib.get("time", "0") or 0) for case in tests)
    suite.attrib.update(
        tests=str(len(tests)),
        failures=str(failures),
        errors=str(errors),
        skipped=str(skipped),
        time=f"{elapsed:.6f}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def _installed_runtime(wheel: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("curobo-metal")
    distribution_root = Path(distribution.locate_file("")).resolve()
    import curobo
    import torch

    import_path = Path(curobo.__file__ or "").resolve()
    if "site-packages" not in import_path.parts:
        raise ValueError(f"curobo is not imported from site-packages: {import_path}")
    try:
        import_path.relative_to(distribution_root)
    except ValueError as exc:
        raise ValueError(f"curobo is outside the installed distribution: {import_path}") from exc
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ValueError("portable replay requires an available PyTorch MPS backend")
    verified_files = _verify_installed_record(distribution_root, wheel)
    return {
        "installed_wheel": True,
        "candidate_wheel_record_verified": True,
        "verified_record_files": verified_files,
        "wheel": {"path": str(wheel), "sha256": _sha256(wheel)},
        "distribution_version": distribution.version,
        "curobo_import_path": str(import_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_enable_mps_fallback": "0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--module-timeout",
        type=float,
        default=DEFAULT_MODULE_TIMEOUT_SECONDS,
        help="Outer wall-clock limit for each isolated pytest module worker.",
    )
    parser.add_argument(
        "--terminate-grace",
        type=float,
        default=DEFAULT_TERMINATE_GRACE_SECONDS,
        help="Seconds between terminating and killing a wedged module worker.",
    )
    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Portable path below curobo/tests; repeat for a bounded implementation wave.",
    )
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "gauntlet/portable-dropin-parity.json"
    )
    parser.add_argument(
        "--execution-census",
        type=Path,
        default=ROOT / "artifacts/api_compat/upstream-execution-census.json",
    )
    args = parser.parse_args(argv)
    if args.module_timeout <= PYTEST_CASE_TIMEOUT_SECONDS:
        parser.error(
            f"--module-timeout must exceed pytest's {PYTEST_CASE_TIMEOUT_SECONDS:g}s "
            "case timeout so pytest can emit its stack capture"
        )
    if args.terminate_grace < 0:
        parser.error("--terminate-grace must be non-negative")

    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        parser.error(f"expected upstream {PINNED_REVISION}, found {revision}")
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")
    try:
        runtime = _installed_runtime(wheel)
    except (ValueError, importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))

    policy = _load(args.policy)
    execution_census = _load(args.execution_census)
    paths = selected_test_paths(execution_census, policy["source_disposition"])
    if args.tests:
        invalid = sorted(set(args.tests) - set(paths))
        if invalid:
            parser.error(f"--test is outside the substituted surface: {invalid}")
        paths = tuple(dict.fromkeys(args.tests))
    test_root = upstream / "curobo/tests"
    source_hashes: dict[str, str] = {}
    args.junit.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="curobo-portable-upstream-") as raw_stage:
        stage = Path(raw_stage)
        staged_tests = stage / "tests"
        staged_tests.mkdir(parents=True)
        staged_helpers = stage / "_curobo_upstream_helpers"
        staged_helpers.mkdir()
        (staged_helpers / "__init__.py").write_text("", encoding="utf-8")
        lidar_helper = upstream / "curobo/examples/reference/lidar_volumetric_mapping.py"
        if not lidar_helper.is_file():
            parser.error(f"missing pinned upstream helper: {lidar_helper}")
        shutil.copyfile(lidar_helper, staged_helpers / lidar_helper.name)
        conftest_source = (test_root / "conftest.py").read_text(encoding="utf-8")
        conftest_adaptation = adapt_source(conftest_source, adapt_availability=False)
        (staged_tests / "conftest.py").write_text(
            conftest_adaptation.source, encoding="utf-8"
        )
        adaptation_counts = {
            "device_string_replacements": conftest_adaptation.device_string_replacements,
            "availability_replacements": conftest_adaptation.availability_replacements,
            "availability_preserved": conftest_adaptation.availability_preserved,
            "helper_import_replacements": conftest_adaptation.helper_import_replacements,
        }
        adaptation_records = {
            "conftest.py": {
                "adapted_sha256": _sha256(staged_tests / "conftest.py"),
                "device_string_replacements": conftest_adaptation.device_string_replacements,
                "availability_replacements": conftest_adaptation.availability_replacements,
                "availability_preserved": conftest_adaptation.availability_preserved,
                "helper_import_replacements": conftest_adaptation.helper_import_replacements,
            }
        }
        for relative in paths:
            source = test_root / relative
            if not source.is_file():
                parser.error(f"missing pinned test: {source}")
            destination = staged_tests / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            module = _module_for_path(relative)
            adaptation = adapt_source(
                source.read_text(encoding="utf-8"),
                preserve_availability_scopes=_raw_mechanism_scopes(policy, module),
            )
            destination.write_text(adaptation.source, encoding="utf-8")
            adaptation_counts["device_string_replacements"] += (
                adaptation.device_string_replacements
            )
            adaptation_counts["availability_replacements"] += (
                adaptation.availability_replacements
            )
            adaptation_counts["availability_preserved"] += adaptation.availability_preserved
            adaptation_counts["helper_import_replacements"] += (
                adaptation.helper_import_replacements
            )
            source_hashes[module] = _sha256(source)
            adaptation_records[module] = {
                "adapted_sha256": _sha256(destination),
                "device_string_replacements": adaptation.device_string_replacements,
                "availability_replacements": adaptation.availability_replacements,
                "availability_preserved": adaptation.availability_preserved,
                "helper_import_replacements": adaptation.helper_import_replacements,
            }
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        shard_root = stage / "junit-shards"
        shard_root.mkdir()
        runs = []
        for index, relative in enumerate(paths):
            module = _module_for_path(relative)
            runs.append(
                _run_module(
                    module=module,
                    test_path=str(Path("tests") / relative),
                    junit_path=shard_root / f"{index:03d}.xml",
                    cwd=stage,
                    environment=environment,
                    timeout_seconds=args.module_timeout,
                    terminate_grace_seconds=args.terminate_grace,
                )
            )
        module_runs = tuple(runs)
        merge_module_junit(module_runs, args.junit.resolve())

    if not args.junit.is_file():
        parser.error(f"module replay did not produce merged JUnit: {args.junit}")

    result = build_census(
        policy=policy,
        execution_census=execution_census,
        junit_path=args.junit.resolve(),
    )
    result["execution"] = runtime
    result["source"] = {
        "upstream_path": str(upstream),
        "conftest_sha256": _sha256(test_root / "conftest.py"),
        "modules": source_hashes,
        "portable_adapter": {
            "scope": (
                "exact device string literals, torch.cuda.is_available gates, and "
                "pinned unshipped test-helper imports"
            ),
            "source_sha256": _sha256(Path(__file__).with_name("portable_test_adapter.py")),
            "records": adaptation_records,
            **adaptation_counts,
        },
    }
    nonzero_codes = [run.returncode for run in module_runs if run.returncode]
    result["pytest_exit_code"] = nonzero_codes[0] if nonzero_codes else 0
    result["pytest_shards"] = [
        {
            "module": run.module,
            "test_path": run.test_path,
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "elapsed_seconds": round(run.elapsed_seconds, 6),
            "junit_complete": run.junit_complete,
        }
        for run in module_runs
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["implementation_parity_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
