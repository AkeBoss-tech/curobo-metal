#!/usr/bin/env python3
"""Fetch, optionally install, and audit the exact cuRoboV2 compatibility pin."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PINNED_REPOSITORY = "https://github.com/NVlabs/curobo.git"
PINNED_SHA = "8e734f3ced1df898990bcd92de40abce475907db"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORT_ROOTS = {"cuda", "isaacsim", "omni", "warp"}
TARGETED_UPSTREAM_APIS = {
    "curobo.kinematics.KinematicsCfg": "curobo/_src/robot/kinematics/kinematics_cfg.py",
    "curobo.kinematics.Kinematics": "curobo/_src/robot/kinematics/kinematics.py",
    "curobo.types.robot.RobotCfg": "curobo/_src/types/robot.py",
    "curobo._src.robot.types.kinematics_params.KinematicsParams": (
        "curobo/_src/robot/types/kinematics_params.py"
    ),
}


def run(*command: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def git(source: Path, *arguments: str, capture: bool = True) -> str:
    return run("git", "-C", str(source), *arguments, capture=capture)


def verify_pin(source: Path) -> dict[str, str]:
    """Verify ref resolution and recompute the Git SHA-1 from commit bytes."""
    resolved = git(source, "rev-parse", "HEAD^{commit}")
    if resolved != PINNED_SHA:
        raise SystemExit(f"expected checked-out commit {PINNED_SHA}, found {resolved}")
    object_type = git(source, "cat-file", "-t", resolved)
    if object_type != "commit":
        raise SystemExit(f"{resolved} is a {object_type}, not a commit")
    commit_bytes = subprocess.check_output(
        ["git", "-C", str(source), "cat-file", "commit", resolved]
    )
    header = f"commit {len(commit_bytes)}\0".encode("ascii")
    recomputed = hashlib.sha1(header + commit_bytes).hexdigest()
    if recomputed != PINNED_SHA:
        raise SystemExit(
            f"commit object failed cryptographic SHA-1 verification: {recomputed}"
        )
    return {
        "repository": PINNED_REPOSITORY,
        "expected_revision": PINNED_SHA,
        "resolved_revision": resolved,
        "recomputed_git_object_sha1": recomputed,
        "commit_subject": git(source, "show", "-s", "--format=%s", resolved),
    }


def fetch(destination: Path) -> dict[str, str]:
    destination = destination.resolve()
    if destination.exists() and not (destination / ".git").is_dir():
        raise SystemExit(f"destination exists but is not a Git checkout: {destination}")
    if not destination.exists():
        destination.mkdir(parents=True)
        run("git", "init", str(destination))
        git(destination, "remote", "add", "origin", PINNED_REPOSITORY, capture=False)
    remotes = git(destination, "remote", "get-url", "origin")
    if remotes != PINNED_REPOSITORY:
        raise SystemExit(f"origin is {remotes!r}, expected {PINNED_REPOSITORY!r}")
    git(destination, "fetch", "--depth", "1", "origin", PINNED_SHA, capture=False)
    git(destination, "checkout", "--detach", PINNED_SHA, capture=False)
    return verify_pin(destination)


def install(source: Path, target: Path, python: str) -> dict[str, str]:
    verified = verify_pin(source.resolve())
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    arguments = [
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(source.resolve()),
    ]
    has_pip = subprocess.run(
        [python, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if has_pip:
        run(python, "-m", "pip", *arguments)
        installer = f"{python} -m pip"
    elif shutil.which("uv"):
        run("uv", "pip", *arguments, "--python", python)
        installer = "uv pip"
    else:
        raise SystemExit(f"{python} has no pip and uv is unavailable")
    return {
        **verified,
        "install_target": str(target),
        "dependencies": "not installed",
        "installer": installer,
    }


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def runtime_import_audit() -> dict[str, Any]:
    code = r"""
import importlib
import json
import sys
before = set(sys.modules)
module = importlib.import_module("curobo_metal.compat")
after = set(sys.modules)
forbidden = sorted({
    name.partition(".")[0]
    for name in after - before
    if name.partition(".")[0] in {"cuda", "isaacsim", "omni", "warp"}
})
print(json.dumps({
    "module": module.__name__,
    "forbidden_new_import_roots": forbidden,
    "torch_loaded": "torch" in after,
}))
"""
    environment = dict(os.environ)
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    output = subprocess.check_output([sys.executable, "-c", code], text=True, env=environment)
    result = json.loads(output)
    if result["forbidden_new_import_roots"] or result["torch_loaded"]:
        raise SystemExit(f"compatibility import audit failed: {result}")
    return result


def device_environment() -> dict[str, Any]:
    code = r"""
import importlib.util
import json
torch_installed = importlib.util.find_spec("torch") is not None
cuda_available = False
if torch_installed:
    import torch
    cuda_available = bool(torch.cuda.is_available())
print(json.dumps({
    "torch_installed": torch_installed,
    "torch_cuda_available": cuda_available,
    "warp_installed": importlib.util.find_spec("warp") is not None,
    "isaacsim_installed": importlib.util.find_spec("isaacsim") is not None,
}))
"""
    return json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))


def audit(source: Path) -> dict[str, Any]:
    source = source.resolve()
    verified = verify_pin(source)
    missing = [
        {"api": api, "path": path}
        for api, path in TARGETED_UPSTREAM_APIS.items()
        if not (source / path).is_file()
    ]
    if missing:
        raise SystemExit(f"pinned public API source files are missing: {missing}")
    compat_files = sorted((REPOSITORY_ROOT / "src/curobo_metal/compat").glob("*.py"))
    static_imports = {
        str(path.relative_to(REPOSITORY_ROOT)): sorted(imported_roots(path))
        for path in compat_files
    }
    forbidden = sorted(
        {
            root
            for roots in static_imports.values()
            for root in roots
            if root in FORBIDDEN_IMPORT_ROOTS or root == "torch"
        }
    )
    if forbidden:
        raise SystemExit(f"compatibility layer has forbidden imports: {forbidden}")
    return {
        "schema_version": 1,
        "pin_verification": verified,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "targeted_upstream_apis": TARGETED_UPSTREAM_APIS,
        "static_compat_imports": static_imports,
        "runtime_import": runtime_import_audit(),
        "device_environment": device_environment(),
        "cuda_required": False,
        "isaac_required": False,
        "warp_required": False,
        "robot_scene_collision_cfg_initialized": False,
        "result": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--destination", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--source", type=Path, required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--source", type=Path, required=True)
    install_parser.add_argument("--target", type=Path, required=True)
    install_parser.add_argument("--python", default=sys.executable)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--source", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "fetch":
        result: dict[str, Any] = fetch(args.destination)
    elif args.command == "verify":
        result = verify_pin(args.source.resolve())
    elif args.command == "install":
        result = install(args.source, args.target, args.python)
    else:
        result = audit(args.source)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
