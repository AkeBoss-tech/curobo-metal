#!/usr/bin/env python3
"""Backend-neutral tensor replay bundle creation and comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

FORMAT = "curobo-metal-parity-replay"
VERSION = 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: {"dtype": str(data[key].dtype), "shape": list(data[key].shape)}
            for key in sorted(data.files)
        }


def pack(args: argparse.Namespace) -> None:
    inputs, outputs = args.inputs.resolve(), args.outputs.resolve()
    manifest = {
        "format": FORMAT, "version": VERSION, "case_id": args.case_id,
        "operation": args.operation, "backend": args.backend,
        "upstream_revision": args.upstream_revision,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "inputs": {"file": inputs.name, "sha256": digest(inputs), "tensors": describe(inputs)},
        "outputs": {"file": outputs.name, "sha256": digest(outputs), "tensors": describe(outputs)},
    }
    args.bundle.mkdir(parents=True, exist_ok=True)
    for source in (inputs, outputs):
        (args.bundle / source.name).write_bytes(source.read_bytes())
    (args.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compare(args: argparse.Namespace) -> None:
    left, right = args.left / "manifest.json", args.right / "manifest.json"
    lm, rm = json.loads(left.read_text()), json.loads(right.read_text())
    for manifest in (lm, rm):
        if (manifest.get("format"), manifest.get("version")) != (FORMAT, VERSION):
            raise SystemExit("unsupported replay manifest")
    if lm["operation"] != rm["operation"] or lm["case_id"] != rm["case_id"]:
        raise SystemExit("operation/case mismatch")
    with np.load(args.left / lm["outputs"]["file"], allow_pickle=False) as a, np.load(args.right / rm["outputs"]["file"], allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise SystemExit(f"output keys differ: {set(a.files) ^ set(b.files)}")
        report = {"case_id": lm["case_id"], "operation": lm["operation"], "left_backend": lm["backend"], "right_backend": rm["backend"], "tensors": {}, "passed": True}
        for key in sorted(a.files):
            if a[key].shape != b[key].shape:
                item = {"passed": False, "reason": "shape", "left": list(a[key].shape), "right": list(b[key].shape)}
            else:
                delta = np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))
                ok = bool(np.allclose(a[key], b[key], rtol=args.rtol, atol=args.atol, equal_nan=True))
                item = {"passed": ok, "max_abs": float(delta.max(initial=0)), "mean_abs": float(delta.mean()) if delta.size else 0.0}
            report["tensors"][key] = item
            report["passed"] &= item["passed"]
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(required=True)
    p = sub.add_parser("pack")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--operation", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--upstream-revision", default=PIN)
    p.set_defaults(func=pack)
    c = sub.add_parser("compare")
    c.add_argument("left", type=Path)
    c.add_argument("right", type=Path)
    c.add_argument("--rtol", type=float, default=1e-5)
    c.add_argument("--atol", type=float, default=1e-6)
    c.set_defaults(func=compare)
    return root


PIN = "8e734f3ced1df898990bcd92de40abce475907db"
if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.func(parsed)
