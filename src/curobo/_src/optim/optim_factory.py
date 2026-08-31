"""Portable optimizer construction for the pinned V2 solver-type contract."""

from __future__ import annotations

from typing import Dict, List

from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
)
from curobo._src.optim.gradient.gradient_descent import (
    GradientDescentOpt,
    GradientDescentOptCfg,
    LineSearchGradientDescentOpt,
)
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
from curobo._src.optim.particle.mppi import MPPI, MPPICfg
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


_SOLVER_CONFIG_MAP = {
    "lbfgs": LBFGSOptCfg,
    "gradient_descent": GradientDescentOptCfg,
    "line_search_gradient_descent": GradientDescentOptCfg,
    "conjugate_gradient": ConjugateGradientOptCfg,
    "lsr1": LBFGSOptCfg,
    "scipy": ScipyOptCfg,
    "torch": TorchOptCfg,
    "mppi": MPPICfg,
    "es": EvolutionStrategiesCfg,
}

_SOLVER_MAP = {
    "lbfgs": LBFGSOpt,
    "gradient_descent": GradientDescentOpt,
    "line_search_gradient_descent": LineSearchGradientDescentOpt,
    "conjugate_gradient": ConjugateGradientOpt,
    "lsr1": LSR1Opt,
    "scipy": ScipyOpt,
    "torch": TorchOpt,
    "mppi": MPPI,
    "es": EvolutionStrategies,
}


def create_optimization_config(config_dict: Dict, device_cfg: DeviceCfg):
    """Construct the configuration selected by its pinned ``solver_type``."""

    solver_type = config_dict.get("solver_type", "")
    config_class = _SOLVER_CONFIG_MAP.get(solver_type)
    if config_class is None:
        raise ValueError(
            f"Unsupported solver type: {solver_type}. "
            f"Available: {list(_SOLVER_CONFIG_MAP.keys())}"
        )
    return config_class(**config_class.create_data_dict(config_dict.copy(), device_cfg))


def create_optimizer(config, rollout: List[Rollout], use_cuda_graph: bool = False):
    """Construct the selected portable optimizer without changing its facade."""

    if not isinstance(rollout, list):
        log_and_raise("rollout must be a list of Rollout")
    optimizer_class = _SOLVER_MAP.get(config.solver_type)
    if optimizer_class is None:
        raise ValueError(
            f"Unsupported solver type: {config.solver_type}. "
            f"Available: {list(_SOLVER_MAP.keys())}"
        )
    return optimizer_class(config=config, rollout_list=rollout, use_cuda_graph=use_cuda_graph)
