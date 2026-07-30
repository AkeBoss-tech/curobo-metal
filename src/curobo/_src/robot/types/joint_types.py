"""Joint type definitions used by the portable robot loader."""

from enum import Enum


class JointType(Enum):
    FIXED = -1
    X_PRISM = 0
    Y_PRISM = 1
    Z_PRISM = 2
    X_ROT = 3
    Y_ROT = 4
    Z_ROT = 5
    X_PRISM_NEG = 6
    Y_PRISM_NEG = 7
    Z_PRISM_NEG = 8
    X_ROT_NEG = 9
    Y_ROT_NEG = 10
    Z_ROT_NEG = 11


__all__ = ["JointType"]
