#!/usr/bin/env python3
"""Run byte-identical portable applications, compare reports, or make a CUDA bundle.

Only the standard library is needed in the orchestration environment. PyTorch
and the selected curobo distribution belong in the isolated target environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

PIN = "8e734f3ced1df898990bcd92de40abce475907db"
MARKER = "CUROBO_APPLICATION_RESULT="
PREFLIGHT = r'''
import base64, csv, hashlib, importlib.metadata as md, io, json, pathlib, sys, zipfile
import torch, curobo
kind, wheel = sys.argv[1:]
owners = md.packages_distributions().get("curobo", [])
assert len(owners) == 1, f"ambiguous curobo namespace owners: {owners}"
dist = md.distribution(owners[0])
root = pathlib.Path(curobo.__file__).resolve().parent
assert "site-packages" in root.parts, f"not an installed package: {root}"
direct = json.loads(dist.read_text("direct_url.json") or "{}")
assert not direct.get("dir_info", {}).get("editable"), "editable installs are excluded"
checked = 0
if kind == "metal":
    assert owners == ["curobo-metal"], owners
    assert wheel, "a candidate wheel is required"
    with zipfile.ZipFile(wheel) as archive:
        record, = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        for name, encoded, size in csv.reader(io.StringIO(archive.read(record).decode())):
            if not encoded:
                continue
            algo, expected = encoded.split("=", 1)
            assert algo == "sha256"
            path = pathlib.Path(dist.locate_file(name)).resolve()
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
            assert digest == expected, f"installed file differs from wheel: {name}"
            checked += 1
    assert checked > 100
else:
    assert owners[0] != "curobo-metal", owners
    assert torch.cuda.is_available(), "CUDA is unavailable"
    # A VCS install records its immutable revision. A source install must be
    # attested against the pinned source tree separately by the runner.
print(json.dumps({"owners": owners, "version": dist.version, "package_root": str(root),
                  "python": sys.version, "torch": torch.__version__, "wheel_files_verified": checked,
                  "direct_url": direct, "mps_available": torch.backends.mps.is_available(),
                  "cuda_available": torch.cuda.is_available()}))
'''
LAUNCH = r'''
import runpy, sys
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2], run_name="__main__")
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_suite(suite: Path) -> dict:
    manifest = read(suite / "manifest.json")
    if manifest["upstream_revision"] != PIN:
        raise ValueError("unexpected upstream revision")
    if not 6 <= len(manifest["cases"]) <= 10:
        raise ValueError("expected 6–10 application cases")
    if {p.name for p in suite.glob("*.py")} != set(manifest["files"]):
        raise ValueError("application Python file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        path = (suite / name).resolve()
        if path.parent != suite.resolve() or digest(path) != expected:
            raise ValueError(f"application source hash mismatch: {name}")
    for case in manifest["cases"]:
        if case["file"] not in manifest["files"]:
            raise ValueError("unhashed application")
    return manifest


def tensor_devices(value):
    if isinstance(value, dict):
        if {"shape", "dtype", "device"} <= value.keys():
            yield value["device"]
        for child in value.values():
            yield from tensor_devices(child)
    elif isinstance(value, list):
        for child in value:
            yield from tensor_devices(child)


def verify_cuda_install(python: str, source: Path, metadata: dict) -> dict:
    """Attest installed Python code against the clean pinned Git source."""
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError("CUDA checkout is not the pinned revision")
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if status.strip():
        raise ValueError("CUDA source checkout contains tracked changes")
    # Python target may be on this machine in another venv; inspect its files
    # through that interpreter, rather than relying on parent sys.path.
    hashes = {str(p.relative_to(source / "curobo")): digest(p)
              for p in (source / "curobo").rglob("*.py")}
    code = "import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*.py')}))"
    installed = json.loads(subprocess.check_output([python, "-I", "-c", code, metadata["package_root"]], text=True))
    mismatches = [name for name, expected in hashes.items() if installed.get(name) != expected]
    if mismatches:
        raise ValueError(f"installed CUDA source differs from pin: {mismatches[:10]}")
    return {"revision": revision, "python_files_verified": len(hashes)}


def run(args) -> int:
    suite = args.suite.resolve()
    manifest = verify_suite(suite)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"):
        environment.pop(key, None)
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    python = str(args.python.absolute())
    wheel = str(args.wheel.resolve()) if args.wheel else ""
    with tempfile.TemporaryDirectory(prefix="curobo-application-run-") as temporary:
        work = Path(temporary)
        preflight = subprocess.run([python, "-I", "-c", PREFLIGHT, args.backend, wheel],
                                   cwd=work, env=environment, capture_output=True, text=True, timeout=60)
        if preflight.returncode:
            raise RuntimeError(preflight.stderr)
        metadata = json.loads(preflight.stdout.splitlines()[-1])
        if args.backend == "cuda":
            if not args.upstream_source:
                raise ValueError("--upstream-source is required to attest the CUDA install")
            metadata["source_attestation"] = verify_cuda_install(python, args.upstream_source.resolve(), metadata)
        report = {"schema_version": 1, "upstream_revision": PIN, "backend": args.backend,
                  "expected_device": args.expect_device, "manifest_sha256": digest(suite / "manifest.json"),
                  "source_hashes": manifest["files"], "runtime": metadata,
                  "wheel_sha256": digest(args.wheel) if args.wheel else None,
                  "mps_fallback_enabled": False, "cases": [], "passed": False}
        for case in manifest["cases"]:
            start = time.perf_counter()
            entry = {"file": case["file"], "passed": False}
            case_work = work / Path(case["file"]).stem
            case_work.mkdir()
            try:
                result = subprocess.run([python, "-I", "-c", LAUNCH, str(suite), str(suite / case["file"])],
                                        cwd=case_work, env=environment, capture_output=True, text=True,
                                        timeout=args.timeout)
                (output / (case["file"] + ".log")).write_text(result.stdout + "\n" + result.stderr)
                entry["returncode"] = result.returncode
                observations = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
                if result.returncode != 0 or len(observations) != 1:
                    raise ValueError("application failed or did not emit exactly one result; see log")
                entry["observations"] = json.loads(observations[0])
                devices = list(tensor_devices(entry["observations"]))
                if not devices or set(devices) != {args.expect_device}:
                    raise ValueError(f"unexpected observed tensor devices: {devices}")
                verify_suite(suite)
                entry["passed"] = True
            except subprocess.TimeoutExpired as exc:
                entry["error"] = f"timed out after {args.timeout}s"
                (output / (case["file"] + ".log")).write_text(str(exc.stdout or "") + str(exc.stderr or ""))
            except (ValueError, OSError) as exc:
                entry["error"] = str(exc)
            entry["elapsed_seconds"] = time.perf_counter() - start
            report["cases"].append(entry)
            write(output / "report.json", report)
            print(f"{case['file']}: {'PASS' if entry['passed'] else 'FAIL'} ({entry['elapsed_seconds']:.2f}s)", flush=True)
        report["passed"] = all(c["passed"] for c in report["cases"])
        write(output / "report.json", report)
    return 0 if report["passed"] else 1


def compare_value(left, right, *, atol: float, rtol: float, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}: type differs"]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [f"{path}: keys differ"]
        tensor_record = {"shape", "dtype", "device"} <= left.keys()
        errors = []
        for key in left:
            if tensor_record and key == "device":
                continue  # Each report's runner validates platform-specific residency.
            errors += compare_value(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
        return errors
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: length differs"]
        return [e for i, (a, b) in enumerate(zip(left, right))
                for e in compare_value(a, b, atol=atol, rtol=rtol, path=f"{path}[{i}]")]
    if isinstance(left, float):
        return [] if math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, abs_tol=atol, rel_tol=rtol) else [f"{path}: {left} != {right}"]
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def compare(args) -> int:
    manifest = verify_suite(args.suite)
    metal, cuda = read(args.metal), read(args.cuda)
    errors = []
    for report, backend, device in ((metal, "metal", "mps"), (cuda, "cuda", "cuda")):
        if report["backend"] != backend or report["expected_device"] != device or not report["passed"]:
            errors.append(f"invalid {backend} report")
        if report["manifest_sha256"] != digest(args.suite / "manifest.json") or report["source_hashes"] != manifest["files"]:
            errors.append(f"{backend} application hashes differ")
        if report["upstream_revision"] != PIN:
            errors.append(f"{backend} revision differs")
        if [c["file"] for c in report["cases"]] != [c["file"] for c in manifest["cases"]]:
            errors.append(f"{backend} case set differs")
        if any(not c["passed"] or not list(tensor_devices(c.get("observations", {}))) or
               set(tensor_devices(c.get("observations", {}))) != {device} for c in report["cases"]):
            errors.append(f"{backend} case status or residency differs")
    if not errors:
        for policy, a, b in zip(manifest["cases"], metal["cases"], cuda["cases"]):
            errors += compare_value(a["observations"], b["observations"], atol=policy["atol"], rtol=policy["rtol"], path=policy["file"])
    write(args.output, {"passed": not errors, "errors": errors, "metal_report_sha256": digest(args.metal), "cuda_report_sha256": digest(args.cuda)})
    print(f"comparison: {'PASS' if not errors else 'FAIL'} ({len(errors)} differences)")
    return bool(errors)


def bundle(args) -> int:
    verify_suite(args.suite)
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.suite, target / "applications", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, target / "run.py")
    if args.metal:
        shutil.copy2(args.metal, target / "metal-report.json")
    (target / "README.md").write_text(f"""# Unchanged cuRobo V2 application replay

These are project-authored portable applications targeting `{PIN}`, not copies
of NVIDIA tutorials. Application and support files must remain byte-identical.
No CUDA result is implied by the included Metal report.

In a separate CUDA environment install the pinned upstream source non-editably
with its required dependencies. Keep the clean pinned checkout for attestation.
Then run from this directory (substitute interpreter and checkout paths):

```sh
python run.py run --suite applications --python /path/to/cuda-venv/bin/python --backend cuda --expect-device cuda --upstream-source /path/to/pinned/curobo --output cuda-results
python run.py compare --suite applications --metal metal-report.json --cuda cuda-results/report.json --output comparison.json
```

The runner verifies installed upstream Python source against the clean pin and
checks source hashes, status, tensor shapes, dtypes, residency and finite values.
Numerical tolerances are declared per application in `manifest.json`. Solver
solutions may differ; only the recorded outcome tensors are numerically compared.
Timings include process startup and are not warm-latency benchmarks. Each process
has a 600-second timeout, configurable with `--timeout`. No application source,
PyTorch API, device default, or assertion is rewritten by this runner.
""")
    hashes = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*")) if p.is_file()}
    write(target / "bundle-hashes.json", hashes)
    print(target)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_suite = Path(__file__).resolve().parents[2] / "examples/application_compat"
    p = sub.add_parser("run")
    p.add_argument("--suite", type=Path, default=default_suite)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--backend", choices=["metal", "cuda"], required=True)
    p.add_argument("--expect-device", choices=["mps", "cpu", "cuda"], required=True)
    p.add_argument("--wheel", type=Path)
    p.add_argument("--upstream-source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.set_defaults(func=run)
    p = sub.add_parser("compare")
    p.add_argument("--suite", type=Path, default=default_suite)
    p.add_argument("--metal", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=compare)
    p = sub.add_parser("bundle")
    p.add_argument("--suite", type=Path, default=default_suite)
    p.add_argument("--metal", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=bundle)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
