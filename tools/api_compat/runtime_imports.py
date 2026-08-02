#!/usr/bin/env python3
"""Import every pinned runtime namespace from a source tree or installed wheel.

The static API inventory deliberately avoids importing either cuRobo package.  This
tool is the complementary runtime check: it reads only the committed inventory and
imports every downstream runtime module listed there.  It never needs an upstream
checkout, and it does not exercise CUDA/Warp-only operations.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"


def runtime_module_names(payload: dict[str, Any]) -> list[str]:
    """Validate and return the unique lexical runtime module list."""
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("revision") != PINNED_REVISION:
        raise ValueError("inventory is not pinned to the expected cuRobo V2 revision")
    modules = payload.get("modules")
    if not isinstance(modules, list):
        raise ValueError("inventory modules must be a list")
    names = []
    for module in modules:
        if not isinstance(module, dict) or module.get("surface") != "runtime":
            continue
        name = module.get("name")
        if not isinstance(name, str) or (name != "curobo" and not name.startswith("curobo.")):
            raise ValueError(f"invalid runtime module name: {name!r}")
        names.append(name)
    names = sorted(set(names))
    summary = payload.get("runtime_summary")
    if not isinstance(summary, dict) or summary.get("modules") != len(names):
        raise ValueError("runtime module count disagrees with inventory summary")
    return names


def audit_runtime_imports(
    names: Iterable[str],
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> dict[str, str]:
    """Return an ordered mapping of modules that failed to import."""
    failures: dict[str, str] = {}
    for name in names:
        try:
            importer(name)
        except Exception as error:  # import failures are reported together
            failures[name] = f"{type(error).__name__}: {error}"
    return failures


def _assert_installed_distribution() -> None:
    """Fail if ``curobo`` resolves outside the installed curobo-metal wheel."""
    distribution = importlib.metadata.distribution("curobo-metal")
    distribution_root = Path(distribution.locate_file("")).resolve()
    package = importlib.import_module("curobo")
    package_file = Path(package.__file__ or "").resolve()
    try:
        package_file.relative_to(distribution_root)
    except ValueError as error:
        raise RuntimeError(
            "curobo was not imported from the installed curobo-metal distribution: "
            f"{package_file} is outside {distribution_root}"
        ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--require-installed-wheel",
        action="store_true",
        help="verify that curobo resolves from installed curobo-metal files",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.inventory.read_text(encoding="utf-8"))
        names = runtime_module_names(payload)
        if args.require_installed_wheel:
            _assert_installed_distribution()
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"runtime import gate setup failed: {error}", file=sys.stderr)
        return 2

    failures = audit_runtime_imports(names)
    if failures:
        print("runtime import gate failed:", file=sys.stderr)
        for name, reason in failures.items():
            print(f"  {name}: {reason}", file=sys.stderr)
        return 1
    print(f"runtime import gate passed: {len(names)} pinned cuRobo runtime modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
