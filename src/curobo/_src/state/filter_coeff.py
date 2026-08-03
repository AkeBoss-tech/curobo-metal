"""Per-component joint-state filter coefficients.

Coefficients are Python scalars by design.  PyTorch promotes them directly on
the operand device, so one configuration can drive CPU or MPS state filters
without hidden host tensors or a CUDA-specific coefficient buffer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FilterCoeff:
    """Blend weights for position and its first three derivatives.

    A zero-initialized record is the pinned V2 default.  It retains the prior
    command on subsequent filter calls; callers opt into a channel explicitly
    by supplying a coefficient (``1.0`` means replace that channel).  Values
    remain deliberately unconstrained because the upstream value object also
    permits extrapolating and negative weights.
    """

    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0
