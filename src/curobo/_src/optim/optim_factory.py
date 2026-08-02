from curobo._src.optim.gradient import LBFGSOpt,LBFGSOptCfg,GradientDescentOpt,GradientDescentOptCfg,ConjugateGradientOpt,ConjugateGradientOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt
from curobo._src.optim.particle.mppi import MPPI,MPPICfg
from curobo._src.optim.particle.evolution_strategies import EvolutionStrategies, EvolutionStrategiesCfg
from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
def create_optimization_config(config_dict,device_cfg):
    name=str(config_dict.get("solver_type",config_dict.get("type","lbfgs"))).lower()
    cls=(EvolutionStrategiesCfg if name in {"es","evolution_strategies","evolution"} else
         ScipyOptCfg if "scipy" in name else TorchOptCfg if "torch" in name else
         MPPICfg if "mppi" in name else ConjugateGradientOptCfg if "conjugate" in name else
         GradientDescentOptCfg if "gradient" in name else LBFGSOptCfg)
    return cls(**cls.create_data_dict(config_dict,device_cfg))
def create_optimizer(config,rollout,use_cuda_graph=False):
    name=str(config.solver_type).lower()
    cls=(EvolutionStrategies if name in {"es","evolution_strategies","evolution"} else
         ScipyOpt if "scipy" in name else TorchOpt if "torch" in name else
         MPPI if "mppi" in name else ConjugateGradientOpt if "conjugate" in name else
         LSR1Opt if "lsr1" in name else GradientDescentOpt if "gradient" in name else LBFGSOpt)
    return cls(config,rollout,use_cuda_graph=use_cuda_graph)
