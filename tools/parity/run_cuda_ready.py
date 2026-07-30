#!/usr/bin/env python3
"""Run every implemented pinned-upstream CUDA adapter and compare to Metal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .compare_paired import compare_ready
from .cuda_adapters import ADAPTERS
from .cuda_runtime import revision
from .replay_registry import PIN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--metal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = revision(args.upstream)
    if actual != PIN:
        raise SystemExit(f"refusing upstream revision {actual}; required {PIN}")
    args.output.mkdir(parents=True, exist_ok=True)
    for capability in sorted(ADAPTERS):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.parity.run_pinned_cuda",
                "--upstream",
                str(args.upstream),
                "--input",
                str(args.metal_root / capability / "inputs.npz"),
                "--capability",
                capability,
                "--output",
                str(args.output / capability),
            ],
            check=True,
        )
    report = compare_ready(args.metal_root, args.output)
    import json

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "paired-report.json").write_text(payload)
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
