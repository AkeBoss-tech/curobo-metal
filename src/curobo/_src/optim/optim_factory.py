from curobo._src.optim.gradient import LBFGSOpt,LBFGSOptCfg,GradientDescentOpt,GradientDescentOptCfg,ConjugateGradientOpt,ConjugateGradientOptCfg
from curobo._src.optim.particle.mppi import MPPI,MPPICfg
def create_optimization_config(config_dict,device_cfg):
    name=str(config_dict.get("solver_type",config_dict.get("type","lbfgs"))).lower()
    cls=MPPICfg if "mppi" in name else ConjugateGradientOptCfg if "conjugate" in name else GradientDescentOptCfg if "gradient" in name else LBFGSOptCfg
    return cls(**cls.create_data_dict(config_dict,device_cfg))
def create_optimizer(config,rollout,use_cuda_graph=False):
    name=str(config.solver_type).lower()
    cls=MPPI if "mppi" in name else ConjugateGradientOpt if "conjugate" in name else GradientDescentOpt if "gradient" in name else LBFGSOpt
    return cls(config,rollout,use_cuda_graph=use_cuda_graph)
