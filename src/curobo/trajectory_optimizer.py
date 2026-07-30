"""Public cuRoboV2 trajectory optimization aliases."""

from curobo._src.solver.solver_trajopt import TrajOptSolver as TrajectoryOptimizer
from curobo._src.solver.solver_trajopt_cfg import (
    TrajOptSolverCfg as TrajectoryOptimizerCfg,
)
from curobo._src.solver.solver_trajopt_result import (
    TrajOptSolverResult as TrajectoryOptimizerResult,
)

__all__ = [
    "TrajectoryOptimizer", "TrajectoryOptimizerCfg", "TrajectoryOptimizerResult",
]
