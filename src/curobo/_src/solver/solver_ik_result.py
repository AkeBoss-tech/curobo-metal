from dataclasses import dataclass
from .solver_base_result import BaseSolverResult


@dataclass
class IKSolverResult(BaseSolverResult):
    pass


__all__ = ["IKSolverResult"]
