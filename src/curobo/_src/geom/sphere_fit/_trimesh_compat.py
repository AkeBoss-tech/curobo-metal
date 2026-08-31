"""Typing-only stand-in for optional trimesh annotations.

Sphere-fit metrics operate on the small mesh protocol used by cuRobo
(``bounds``, ``vertices``, and ``volume``), so importing the full optional
dependency is unnecessary at runtime.
"""

from typing import Any

Trimesh = Any

