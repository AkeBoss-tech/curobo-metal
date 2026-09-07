"""Established reacher solve-state names."""

from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState

ReacherSolveState = SolveState
ReacherSolveType = SolveMode
__all__ = ["ReacherSolveState", "ReacherSolveType"]
