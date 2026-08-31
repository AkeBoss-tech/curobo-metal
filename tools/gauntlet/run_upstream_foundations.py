#!/usr/bin/env python3
"""Run an unchanged upstream-test batch against an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
TESTS = (
    "test_curobo_version.py",
    "_src/cost/test_cost_types.py",
    "_src/types/test_control_space.py",
    "_src/solver/test_solve_mode.py",
    "_src/state/test_filter_coeff.py",
    "_src/util/test_logging.py",
    "_src/util/test_python_util.py",
    "_src/types/test_content_path.py",
    "_src/util/test_config_io.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Pinned path below curobo/tests (repeatable; defaults to the foundation batch).",
    )
    args = parser.parse_args(argv)
    selected_tests = tuple(args.tests or TESTS)
    if not selected_tests or any(
        not item.endswith(".py") or Path(item).is_absolute() or ".." in Path(item).parts
        for item in selected_tests
    ):
        parser.error("--test values must be safe relative Python paths")

    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        parser.error(f"expected upstream {PINNED_REVISION}, found {revision}")
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")

    distribution = importlib.metadata.distribution("curobo-metal")
    distribution_root = Path(distribution.locate_file("")).resolve()
    import curobo
    import pytest
    import torch

    curobo_path = Path(curobo.__file__ or "").resolve()
    if "site-packages" not in curobo_path.parts:
        parser.error(f"curobo is not imported from site-packages: {curobo_path}")
    try:
        curobo_path.relative_to(distribution_root)
    except ValueError:
        parser.error(f"curobo is outside the installed distribution: {curobo_path}")

    test_root = upstream / "curobo/tests"
    source_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="curobo-upstream-foundations-") as raw_stage:
        stage = Path(raw_stage)
        conftest = test_root / "conftest.py"
        shutil.copy2(conftest, stage / "conftest.py")
        for relative in selected_tests:
            source = test_root / relative
            if not source.is_file():
                parser.error(f"missing pinned test: {source}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_hashes[f"curobo.tests.{relative[:-3].replace('/', '.')}"] = _sha256(source)

        junit = stage / "junit.xml"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "--import-mode=importlib",
            f"--junitxml={junit}",
            *selected_tests,
        ]
        completed = subprocess.run(
            command, cwd=stage, env=environment, capture_output=True, text=True
        )
        root = ET.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            parser.error("pytest did not produce a testsuite record")
        counts = {
            name: int(suite.attrib.get(name, "0"))
            for name in ("tests", "failures", "errors", "skipped")
        }
        passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        result = {
            "schema_version": 1,
            "upstream_revision": revision,
            "wheel": {
                "path": str(wheel),
                "sha256": _sha256(wheel),
                "version": distribution.version,
                "import_path": str(curobo_path),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "pytorch_enable_mps_fallback": "0",
            },
            "pytest": {**counts, "passed": passed, "exit_code": completed.returncode},
            "conftest_sha256": _sha256(conftest),
            "modules": source_hashes,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if completed.returncode or counts["failures"] or counts["errors"] or counts["skipped"]:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return 1
        print(f"unchanged upstream foundations passed: {passed} tests")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
