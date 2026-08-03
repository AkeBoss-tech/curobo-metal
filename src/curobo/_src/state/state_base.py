"""Sequence base used by portable cuRobo state value objects.

The pinned V2 type is deliberately an empty dataclass: concrete state records
provide indexing and length while this base supplies the common ``Sequence``
and dataclass identity.  Keeping that small contract matters for downstream
code which uses :func:`dataclasses.is_dataclass` and ``isinstance`` checks
without requiring any CUDA-backed state buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class State(Sequence):
    """Abstract, device-neutral base for sequence-shaped state records."""

    pass
