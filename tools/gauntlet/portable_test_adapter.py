"""Auditable, device-only adaptations for pinned upstream test definitions."""

from __future__ import annotations

import ast
import io
import textwrap
import tokenize
from dataclasses import dataclass


DEVICE_STRINGS = {"cuda": "mps", "cuda:0": "mps:0"}
CUDA_AVAILABLE = "torch.cuda.is_available()"
MPS_AVAILABLE = "torch.backends.mps.is_available()"
CUDA_SYNCHRONIZE = "torch.cuda.synchronize()"
MPS_SYNCHRONIZE = "torch.mps.synchronize()"
UPSTREAM_HELPER_IMPORTS = {
    "from curobo.examples.reference.lidar_volumetric_mapping import ": (
        "from _curobo_upstream_helpers.lidar_volumetric_mapping import "
    )
}


@dataclass(frozen=True)
class Adaptation:
    source: str
    device_string_replacements: int
    availability_replacements: int
    availability_preserved: int
    helper_import_replacements: int
    is_cuda_assertion_replacements: int
    synchronize_replacements: int
    oracle_replacements: int
    mechanism_scaffold_guards: int


VOXEL_COLLISION_MODULE = "curobo.tests._src.geom.sdf.test_voxel_collision"
VOXEL_WARP_SCAFFOLD_START = "for _module_path in OBSTACLE_SDF_MODULES:\n"
VOXEL_WARP_SCAFFOLD_END = "\ndef _make_empty_esdf(\n"


def _guard_voxel_collision_mechanism_scaffold(source: str) -> tuple[str, int]:
    """Keep raw Warp declarations from blocking adjacent portable case collection.

    The pinned module registers foreign Warp functions and decorates test-only
    kernels at import time, before pytest can apply its per-case CUDA skips.
    On the portable wheel those foreign functions deliberately expose no Warp
    ABI, so registration fails during collection.  The guarded block is used
    exclusively by reviewed raw-kernel cases; the four VoxelData construction
    cases below it remain unchanged and executable on MPS.
    """

    start = source.find(VOXEL_WARP_SCAFFOLD_START)
    end = source.find(VOXEL_WARP_SCAFFOLD_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("pinned voxel-collision Warp scaffold shape changed")
    scaffold = source[start:end]
    guarded = "if torch.cuda.is_available():\n" + textwrap.indent(scaffold, "    ")
    return source[:start] + guarded + source[end:], 1


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_availability_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and _dotted_name(node.func) == "torch.cuda.is_available"
    )


def _scope_name(scope: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    if isinstance(scope, ast.ClassDef):
        return scope.name
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "<module>"
    parent = parents.get(scope)
    if isinstance(parent, ast.ClassDef):
        return f"{parent.name}.{scope.name}"
    return scope.name


def _availability_spans(
    source: str, preserve_availability_scopes: frozenset[str]
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Return portable availability-call spans and the total call count."""

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    spans = []
    total = 0
    for node in ast.walk(tree):
        if not _is_availability_call(node):
            continue
        total += 1
        scope: ast.AST = node
        while scope in parents and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            scope = parents[scope]
        scope_name = _scope_name(scope, parents)
        if scope_name in preserve_availability_scopes or (
            isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            and scope.name in preserve_availability_scopes
        ):
            continue
        spans.append(
            (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        )
    return spans, total


def _replace_spans(source: str, spans: list[tuple[int, int, int, int]]) -> str:
    lines = source.splitlines(keepends=True)
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)
    replacements = []
    for start_line, start_col, end_line, end_col in spans:
        start = offsets[start_line - 1] + start_col
        end = offsets[end_line - 1] + end_col
        if source[start:end] != CUDA_AVAILABLE:
            raise ValueError("unexpected source span for CUDA availability call")
        replacements.append((start, end))
    for start, end in sorted(replacements, reverse=True):
        source = source[:start] + MPS_AVAILABLE + source[end:]
    return source


def _synchronize_spans(
    source: str, preserve_scopes: frozenset[str]
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Return timing-barrier calls that should target the adapted MPS device."""

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    spans = []
    total = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and _dotted_name(node.func) == "torch.cuda.synchronize"
        ):
            continue
        total += 1
        scope: ast.AST = node
        while scope in parents and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            scope = parents[scope]
        scope_name = _scope_name(scope, parents)
        if scope_name in preserve_scopes or (
            isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            and scope.name in preserve_scopes
        ):
            continue
        spans.append(
            (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        )
    return spans, total


def _replace_synchronize_spans(
    source: str, spans: list[tuple[int, int, int, int]]
) -> str:
    lines = source.splitlines(keepends=True)
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)
    replacements = []
    for start_line, start_col, end_line, end_col in spans:
        start = offsets[start_line - 1] + start_col
        end = offsets[end_line - 1] + end_col
        if source[start:end] != CUDA_SYNCHRONIZE:
            raise ValueError("unexpected CUDA synchronize source span")
        replacements.append((start, end))
    for start, end in sorted(replacements, reverse=True):
        source = source[:start] + MPS_SYNCHRONIZE + source[end:]
    return source


def _is_cuda_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "is_cuda"


def _is_assertion_attribute(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Return whether an ``.is_cuda`` attribute belongs to an ``assert`` test."""

    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.Assert):
            return any(candidate is node for candidate in ast.walk(parent.test))
        # Do not reinterpret ordinary production/test expressions merely because
        # they happen to be nested in a function that also contains an assert.
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        parent = parents.get(parent)
    return False


def _is_cuda_assertion_spans(
    source: str,
) -> list[tuple[int, int, int, int, str]]:
    """Return exact ``.is_cuda`` spans nested in assertion expressions.

    The adapter intentionally does not rewrite arbitrary CUDA predicates or
    production code.  These are only device-residency assertions in tests whose
    CUDA device literals and availability gates have already been substituted
    for MPS.
    """

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    spans = []
    for node in ast.walk(tree):
        if not _is_cuda_attribute(node) or not _is_assertion_attribute(node, parents):
            continue
        segment = ast.get_source_segment(source, node)
        if not segment or not segment.endswith(".is_cuda"):
            raise ValueError("unexpected source span for CUDA residency assertion")
        spans.append(
            (
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
                segment,
            )
        )
    return spans


def _replace_is_cuda_assertions(
    source: str,
    spans: list[tuple[int, int, int, int, str]],
) -> str:
    lines = source.splitlines(keepends=True)
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)
    replacements = []
    for start_line, start_col, end_line, end_col, segment in spans:
        start = offsets[start_line - 1] + start_col
        end = offsets[end_line - 1] + end_col
        if source[start:end] != segment or not segment.endswith(".is_cuda"):
            raise ValueError("unexpected source span for CUDA residency assertion")
        base = segment[: -len(".is_cuda")]
        # Parenthesize the replacement so compound assertions such as
        # ``assert value.is_cuda is True`` retain their original precedence.
        replacements.append((start, end, f"({base}.device.type == 'mps')"))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _adapt_device_cfg_oracles(source: str) -> tuple[str, int]:
    """Adapt three assertions invalidated by portable device normalization.

    The pinned CUDA class defaults to ``cuda:0`` and preserves explicit index
    zero.  The portable class defaults to the available MPS/CPU device and treats an
    omitted CPU/MPS index as operationally equivalent to index zero.  Restrict
    these oracle changes to their exact test functions; ordinary device
    assertions, constructors, and production expressions remain untouched.
    """

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    replacements: list[tuple[int, int, str]] = []
    lines = source.splitlines(keepends=True)
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line)
    expected_scopes = {
        "test_default_initialization": "assert cfg.device == torch.device('mps', 0)",
        "test_from_basic_cpu": 'assert cfg.device == torch.device("cpu", 0)',
        "test_from_basic_cuda": "assert cfg.device == torch.device('mps', 0)",
    }
    replacement_by_scope = {
        "test_default_initialization": "assert cfg.device == torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')",
        "test_from_basic_cpu": (
            'assert cfg.is_same_torch_device(torch.device("cpu", 0))'
        ),
        "test_from_basic_cuda": (
            "assert cfg.is_same_torch_device(torch.device('mps', 0))"
        ),
    }
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        scope = parents.get(node)
        while scope is not None and not isinstance(scope, ast.FunctionDef):
            scope = parents.get(scope)
        if not isinstance(scope, ast.FunctionDef) or scope.name not in expected_scopes:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        segment = source[start:end]
        if segment != expected_scopes[scope.name]:
            continue
        replacements.append((start, end, replacement_by_scope[scope.name]))
        seen.add(scope.name)
    if seen != set(expected_scopes):
        missing = sorted(set(expected_scopes) - seen)
        raise ValueError(f"pinned DeviceCfg oracle shape changed: {missing}")
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source, len(replacements)


def adapt_source(
    source: str,
    *,
    adapt_availability: bool = True,
    preserve_availability_scopes: frozenset[str] = frozenset(),
    test_module: str | None = None,
) -> Adaptation:
    """Replace portable device gates and their exact residency assertions.

    CUDA graph, stream, event, and Warp APIs are intentionally untouched: those
    need case-level mechanism review rather than a broad textual substitution.
    A zero-argument ``torch.cuda.synchronize()`` used only as a device timing
    barrier follows the substituted tensor device, except in those reviewed
    mechanism scopes.
    ``adapt_availability=False`` is the conftest mode and also leaves CUDA
    residency assertions untouched because that file's CUDA guard is intentional.
    """

    output = []
    device_count = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, str) and value in DEVICE_STRINGS:
                token = tokenize.TokenInfo(
                    token.type,
                    repr(DEVICE_STRINGS[value]),
                    token.start,
                    token.end,
                    token.line,
                )
                device_count += 1
        output.append(token)
    adapted = tokenize.untokenize(output)
    helper_import_count = 0
    for upstream_import, staged_import in UPSTREAM_HELPER_IMPORTS.items():
        replacements = adapted.count(upstream_import)
        adapted = adapted.replace(upstream_import, staged_import)
        helper_import_count += replacements
    availability_count = 0
    availability_preserved = 0
    if adapt_availability:
        spans, total = _availability_spans(adapted, preserve_availability_scopes)
        availability_count = len(spans)
        availability_preserved = total - availability_count
        adapted = _replace_spans(adapted, spans)
    synchronize_count = 0
    if adapt_availability:
        synchronize_spans, _total = _synchronize_spans(
            adapted, preserve_availability_scopes
        )
        synchronize_count = len(synchronize_spans)
        adapted = _replace_synchronize_spans(adapted, synchronize_spans)
    is_cuda_assertion_count = 0
    if adapt_availability:
        is_cuda_assertion_spans = _is_cuda_assertion_spans(adapted)
        is_cuda_assertion_count = len(is_cuda_assertion_spans)
        adapted = _replace_is_cuda_assertions(adapted, is_cuda_assertion_spans)
    oracle_count = 0
    if test_module == "curobo.tests._src.types.test_device_cfg":
        adapted, oracle_count = _adapt_device_cfg_oracles(adapted)
    mechanism_scaffold_count = 0
    if test_module == VOXEL_COLLISION_MODULE:
        adapted, mechanism_scaffold_count = _guard_voxel_collision_mechanism_scaffold(
            adapted
        )
    return Adaptation(
        adapted,
        device_count,
        availability_count,
        availability_preserved,
        helper_import_count,
        is_cuda_assertion_count,
        synchronize_count,
        oracle_count,
        mechanism_scaffold_count,
    )
