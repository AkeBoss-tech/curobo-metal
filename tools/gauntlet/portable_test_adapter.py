"""Auditable, device-only adaptations for pinned upstream test definitions."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass


DEVICE_STRINGS = {"cuda": "mps", "cuda:0": "mps:0"}
CUDA_AVAILABLE = "torch.cuda.is_available()"
MPS_AVAILABLE = "torch.backends.mps.is_available()"
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
        if (
            scope_name in preserve_availability_scopes
            or (
                isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                and scope.name in preserve_availability_scopes
            )
        ):
            continue
        spans.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
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


def _is_cuda_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "is_cuda"


def _is_assertion_attribute(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
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
        spans.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, segment))
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


def adapt_source(
    source: str,
    *,
    adapt_availability: bool = True,
    preserve_availability_scopes: frozenset[str] = frozenset(),
) -> Adaptation:
    """Replace portable device gates and their exact residency assertions.

    CUDA graph, stream, event, and Warp APIs are intentionally untouched: those
    need case-level mechanism review rather than a broad textual substitution.
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
    is_cuda_assertion_count = 0
    if adapt_availability:
        is_cuda_assertion_spans = _is_cuda_assertion_spans(adapted)
        is_cuda_assertion_count = len(is_cuda_assertion_spans)
        adapted = _replace_is_cuda_assertions(adapted, is_cuda_assertion_spans)
    return Adaptation(
        adapted,
        device_count,
        availability_count,
        availability_preserved,
        helper_import_count,
        is_cuda_assertion_count,
    )
