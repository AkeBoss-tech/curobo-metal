#!/usr/bin/env python3
"""Build and check a deterministic, import-free cuRobo API inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PINNED_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
UPSTREAM_REPOSITORY = "https://github.com/NVlabs/curobo.git"
SCHEMA_VERSION = 2


def _git(source: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{source} is not a readable Git checkout") from error


def verify_revision(source: Path) -> None:
    revision = _git(source, "rev-parse", "HEAD^{commit}")
    if revision != PINNED_REVISION:
        raise ValueError(
            f"wrong upstream revision: expected {PINNED_REVISION}, found {revision}"
        )


def _module_name(package: Path, path: Path) -> str:
    relative = path.relative_to(package.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _source(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults.extend(args.defaults)
    parameters = []
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        parameters.append(
            {
                "name": argument.arg,
                "kind": "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword",
                "annotation": _source(argument.annotation),
                "default": _source(default),
            }
        )
    if args.vararg:
        parameters.append(
            {"name": args.vararg.arg, "kind": "var_positional", "annotation": _source(args.vararg.annotation), "default": None}
        )
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            {"name": argument.arg, "kind": "keyword_only", "annotation": _source(argument.annotation), "default": _source(default)}
        )
    if args.kwarg:
        parameters.append(
            {"name": args.kwarg.arg, "kind": "var_keyword", "annotation": _source(args.kwarg.annotation), "default": None}
        )
    return {"parameters": parameters, "returns": _source(node.returns), "async": isinstance(node, ast.AsyncFunctionDef)}


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    return [_source(item) or "" for item in node.decorator_list]


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(name == "dataclass" or name.startswith("dataclass(") or name.endswith(".dataclass") for name in _decorator_names(node))


def _is_enum(node: ast.ClassDef) -> bool:
    return any((_source(base) or "").split(".")[-1] in {"Enum", "IntEnum", "Flag", "IntFlag", "StrEnum"} for base in node.bases)


def _public_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name) and not target.id.startswith("_"):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from _public_names(item)


def _class_symbol(node: ast.ClassDef) -> dict[str, Any]:
    dataclass = _is_dataclass(node)
    enum = _is_enum(node)
    members: list[dict[str, Any]] = []
    constructor = None
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            constructor = _signature(child)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
            members.append({"name": child.name, "kind": "method", "signature": _signature(child), "decorators": _decorator_names(child)})
        elif isinstance(child, ast.AnnAssign):
            for name in _public_names(child.target):
                members.append(
                    {
                        "name": name,
                        "kind": "enum_member" if enum else "field",
                        "annotation": _source(child.annotation),
                        "default": _source(child.value),
                    }
                )
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                for name in _public_names(target):
                    members.append({"name": name, "kind": "enum_member" if enum else "attribute", "annotation": None, "default": _source(child.value)})
    return {
        "name": node.name,
        "kind": "class",
        "bases": [_source(base) for base in node.bases],
        "decorators": _decorator_names(node),
        "dataclass": dataclass,
        "dataclass_options": next(
            (
                name
                for name in _decorator_names(node)
                if name == "dataclass"
                or name.startswith("dataclass(")
                or name.endswith(".dataclass")
                or ".dataclass(" in name
            ),
            None,
        ),
        "enum": enum,
        "constructor": constructor,
        "members": sorted(members, key=lambda item: (item["name"], item["kind"])),
    }


def scan_module(package: Path, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    symbols: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            symbols.append({"name": node.name, "kind": "function", "signature": _signature(node), "decorators": _decorator_names(node)})
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            symbols.append(_class_symbol(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _public_names(target):
                    symbols.append(
                        {
                            "name": name,
                            "kind": "assignment",
                            "annotation": _source(node.annotation) if isinstance(node, ast.AnnAssign) else None,
                            "default": _source(node.value),
                        }
                    )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[-1]
                if not name.startswith("_") and name != "*":
                    symbols.append(
                        {
                            "name": name,
                            "kind": "reexport",
                            "source": (
                                ("." * node.level + (node.module or "") + "." + alias.name)
                                if isinstance(node, ast.ImportFrom)
                                else alias.name
                            ),
                        }
                    )
    return {
        "name": _module_name(package, path),
        "path": path.relative_to(package.parent).as_posix(),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "symbols": sorted(symbols, key=lambda item: (item["name"], item["kind"])),
    }


def _local_candidates(local_root: Path, module: str) -> list[Path]:
    suffix = module.removeprefix("curobo").lstrip(".").replace(".", "/")
    candidates = [local_root / "curobo" / suffix, local_root / "curobo_metal" / suffix]
    paths = []
    for candidate in candidates:
        paths.extend([candidate.with_suffix(".py"), candidate / "__init__.py"])
    return paths


def classify_local(module: dict[str, Any], local_root: Path) -> dict[str, Any]:
    path = next((item for item in _local_candidates(local_root, module["name"]) if item.is_file()), None)
    if path is None:
        return {"status": "missing_module", "path": None, "resolved_symbols": 0, "missing_symbols": len(module["symbols"])}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(name for target in targets for name in _public_names(target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[-1]
                for alias in node.names
                if (alias.asname or alias.name.split(".")[-1]) != "*"
            )
    expected = {item["name"] for item in module["symbols"]}
    resolved = len(expected & names)
    return {
        "status": "resolved" if resolved == len(expected) else "partial",
        "path": path.relative_to(local_root).as_posix(),
        "resolved_symbols": resolved,
        "missing_symbols": len(expected - names),
    }


def classify_surface(module_name: str) -> str:
    """Separate downstream runtime API from upstream's bundled validation code."""
    if module_name == "curobo.tests" or module_name.startswith("curobo.tests."):
        return "bundled_test"
    if module_name == "curobo.examples" or module_name.startswith("curobo.examples."):
        return "bundled_example"
    return "runtime"


def _counts(modules: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "modules": len(modules),
        "symbols": sum(len(item["symbols"]) for item in modules),
        "resolved_modules": sum(item["local"]["status"] == "resolved" for item in modules),
        "partial_modules": sum(item["local"]["status"] == "partial" for item in modules),
        "missing_modules": sum(item["local"]["status"] == "missing_module" for item in modules),
    }


def build_inventory(source: Path, local_root: Path) -> dict[str, Any]:
    source = source.resolve()
    verify_revision(source)
    package = source / "curobo"
    if not package.is_dir():
        raise ValueError(f"missing upstream package directory: {package}")
    modules = [
        scan_module(package, path)
        for path in sorted(package.rglob("*.py"), key=lambda item: item.relative_to(package).as_posix())
    ]
    for module in modules:
        module["surface"] = classify_surface(module["name"])
        module["local"] = classify_local(module, local_root.resolve())
    counts = _counts(modules)
    runtime_counts = _counts([item for item in modules if item["surface"] == "runtime"])
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": PINNED_REVISION},
        "method": {"parser": "python-ast", "imports_executed": False, "public_rule": "names not beginning with underscore"},
        "summary": counts,
        "runtime_summary": runtime_counts,
        "modules": modules,
    }


def _encoded(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("src"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="fail if output differs; do not write")
    parser.add_argument("--require-compatible", action="store_true", help="also fail if any module is unresolved")
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory(args.source, args.local_root)
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    encoded = _encoded(inventory)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != encoded:
            print(f"API inventory is stale: regenerate {args.output}", file=sys.stderr)
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.require_compatible and (
        inventory["runtime_summary"]["partial_modules"]
        or inventory["runtime_summary"]["missing_modules"]
    ):
        print("API compatibility gate failed: unresolved upstream modules remain", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
