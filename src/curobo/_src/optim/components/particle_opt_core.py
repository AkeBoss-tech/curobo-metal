from enum import Enum
from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.particle.particle_opt_utils import SquashType,gaussian_entropy,scale_ctrl
class SampleMode(Enum): MEAN="MEAN"; BEST="BEST"; SAMPLE="SAMPLE"
class ParticleOptCore(GradientOptCore):
    def sample_actions(self,init_act): return init_act
    def update_seed(self,init_act): self.seed_action=init_act
    def get_rollouts(self): return getattr(self,"rollouts",None)
    def reset_distribution(self,reset_problem_ids=None): del reset_problem_ids; self.reset()
    def initialize_samples(self): return None
    def update_samples(self): return None
    def update_init_mean(self, init_mean):
        if hasattr(self.config, "init_mean"):
            self.config.init_mean = init_mean
        return init_mean
__all__=["SampleMode","SquashType","gaussian_entropy","scale_ctrl","ParticleOptCore"]
