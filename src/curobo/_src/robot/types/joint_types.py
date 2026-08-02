"""Axis-aligned joint type definitions used by the portable robot loader."""

from __future__ import annotations

from enum import Enum


class JointType(Enum):
    """Pinned V2 joint encodings for fixed, prism, and revolute joints.

    The numerical values form part of serialized kinematic maps, so they are
    intentionally stable.  Arbitrary-axis joints are not represented by this
    CUDA-originated enum; portable URDF loading raises an explicit error for
    them rather than silently choosing a nearest axis.
    """
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
