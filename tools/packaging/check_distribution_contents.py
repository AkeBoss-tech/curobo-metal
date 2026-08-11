#!/usr/bin/env python3
"""Validate release archives contain notices and no generated-tree debris."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


FORBIDDEN_PARTS = {
    ".DS_Store",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"unsupported distribution archive: {path}")


def validate_members(path: Path, members: list[str]) -> list[str]:
    errors: list[str] = []
    paths = [Path(name) for name in members if name and not name.endswith("/")]

    for member in paths:
        if FORBIDDEN_PARTS.intersection(member.parts) or member.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"generated or stale path included: {member.as_posix()}")

    if path.suffix == ".whl":
        licenses = [p for p in paths if ".dist-info" in p.as_posix() and "licenses" in p.parts]
        if not any(p.name == "LICENSE" for p in licenses):
            errors.append("wheel is missing the project LICENSE in dist-info/licenses")
        if not any(p.name == "THIRD_PARTY_NOTICES.md" for p in licenses):
            errors.append("wheel is missing THIRD_PARTY_NOTICES.md in dist-info/licenses")
        forbidden_roots = {"artifacts", "benchmarks", "contracts", "docs", "examples", "tests", "tools"}
        leaked = sorted(p.as_posix() for p in paths if p.parts and p.parts[0] in forbidden_roots)
        errors.extend(f"non-runtime tree leaked into wheel: {name}" for name in leaked)
    else:
        roots = {p.parts[0] for p in paths if p.parts}
        if len(roots) != 1:
            errors.append(f"sdist must have exactly one root directory, found {sorted(roots)}")
        root = next(iter(roots), "")
        for required in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "pyproject.toml"):
            expected = f"{root}/{required}"
            if expected not in members:
                errors.append(f"sdist is missing {required}")

    return errors


def validate_archive(path: Path) -> None:
    errors = validate_members(path, archive_members(path))
    if errors:
        raise SystemExit("\n".join(f"{path.name}: {error}" for error in errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        validate_archive(archive)
        print(f"validated distribution contents: {archive}")


if __name__ == "__main__":
    main()
