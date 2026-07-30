"""Portable implementation of the pinned SciPy optimizer import path."""

from dataclasses import dataclass, field

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer


@dataclass
class ScipyOptCfg(PortableOptCfg):
    solver_type: str = "scipy"
    solver_name: str = "scipy"
    scipy_minimize_method: str = "SLSQP"
    scipy_minimize_kwargs: dict = field(default_factory=dict)
    use_float64_on_cpu: bool = False

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.scipy_minimize_method == "SLSQP":
            self.use_float64_on_cpu = True


class ScipyOpt(PortableOptimizer):
    strategy = "lbfgs"


CudaGraphScipyOpt = ScipyOpt
__all__ = ["ScipyOpt", "ScipyOptCfg", "CudaGraphScipyOpt"]
