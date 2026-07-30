"""Per-component joint-state filter coefficients."""

from dataclasses import dataclass


@dataclass
class FilterCoeff:
    position: float = 1.0
    velocity: float = 1.0
    acceleration: float = 1.0
    jerk: float = 1.0
