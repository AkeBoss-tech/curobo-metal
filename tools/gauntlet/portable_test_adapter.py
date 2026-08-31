"""Auditable, device-only adaptations for pinned upstream test definitions."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass


DEVICE_STRINGS = {"cuda": "mps", "cuda:0": "mps:0"}
CUDA_AVAILABLE = "torch.cuda.is_available()"
MPS_AVAILABLE = "torch.backends.mps.is_available()"


@dataclass(frozen=True)
class Adaptation:
    source: str
    device_string_replacements: int
    availability_replacements: int


def adapt_source(source: str, *, adapt_availability: bool = True) -> Adaptation:
    """Replace device literals and availability gates without renaming CUDA APIs.

    CUDA graph, stream, event, and Warp APIs are intentionally untouched: those
    need case-level mechanism review rather than a broad textual substitution.
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
    availability_count = adapted.count(CUDA_AVAILABLE) if adapt_availability else 0
    if adapt_availability:
        adapted = adapted.replace(CUDA_AVAILABLE, MPS_AVAILABLE)
    return Adaptation(adapted, device_count, availability_count)
