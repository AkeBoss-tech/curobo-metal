"""Deprecated pinned path re-exporting the pose implementation."""

from .pose import Pose
from curobo._src.util.logging import log_warn

__all__ = ["Pose"]
