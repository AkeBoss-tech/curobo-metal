"""Drop-in public optimizer surface for CPU and Apple Metal."""

from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.optim.particle.evolution_strategies import EvolutionStrategies, EvolutionStrategiesCfg
from curobo._src.optim.particle.mppi import MPPI, MPPICfg

__all__ = [
    "EvolutionStrategies", "EvolutionStrategiesCfg", "LBFGSOpt", "LBFGSOptCfg",
    "MPPI", "MPPICfg", "MultiStageOptimizer", "ScipyOpt", "ScipyOptCfg",
    "TorchOpt", "TorchOptCfg",
]
