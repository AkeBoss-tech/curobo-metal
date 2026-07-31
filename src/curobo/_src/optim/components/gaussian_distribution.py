from enum import Enum
import torch
from curobo._src.optim.particle.sample_strategies import MixedParticleSampler
class CovType(Enum): SIGMA_I="SIGMA_I"; DIAG_A="DIAG_A"
class GaussianDistribution:
    def __init__(self,device_cfg,action_horizon,action_dim,cov_type,init_mean,init_cov,sample_params,random_mean=False,seed=0):
        self.device_cfg=device_cfg; self.action_horizon=action_horizon; self.action_dim=action_dim; self.cov_type=cov_type
        self.init_mean=init_mean; self.init_cov=init_cov; self.random_mean=random_mean; self.seed=seed
        self.sample_lib=MixedParticleSampler(sample_params,action_horizon,action_dim); self._sample_set=self._sample_iter=None
        self.mean=self.cov=self.scale_tril=self.inv_cov=None
    def reset_mean(self,num_problems,reset_problem_ids=None):
        value=self.init_mean.to(device=self.device_cfg.device,dtype=self.device_cfg.dtype)
        value=value.expand(num_problems,*value.shape[-2:]).clone()
        if reset_problem_ids is None or self.mean is None:self.mean=value
        else:self.mean[reset_problem_ids]=value[:len(reset_problem_ids)]
    def reset_covariance(self,num_problems):
        cov=torch.as_tensor(self.init_cov,device=self.device_cfg.device,dtype=self.device_cfg.dtype)
        if cov.ndim==0:cov=cov.expand(self.action_dim)
        self.cov=cov.expand(num_problems,self.action_dim).clone(); self.scale_tril=self.cov.sqrt(); self.inv_cov=self.cov.reciprocal()
    def reset(self,num_problems,reset_problem_ids=None):self.reset_mean(num_problems,reset_problem_ids);self.reset_covariance(num_problems)
    def update_mean(self,new_mean,num_problems):self.mean=new_mean.expand(num_problems,*new_mean.shape[-2:]).clone()
    def update_cov_scale(self,new_cov):self.cov=new_cov;self.scale_tril=new_cov.sqrt();self.inv_cov=new_cov.reciprocal()
    def initialize_samples(self,num_problems,sampled_particles_per_problem,num_iters,fixed_samples,sample_per_problem):self.update_samples(num_problems,sampled_particles_per_problem,num_iters,fixed_samples,sample_per_problem)
    def update_samples(self,num_problems,sampled_particles_per_problem,num_iters,fixed_samples,sample_per_problem):
        del fixed_samples,sample_per_problem
        self._sample_set=self.sample_lib.get_samples([num_problems*sampled_particles_per_problem])
        self._sample_iter=self._sample_set.unsqueeze(0).expand(num_iters,*self._sample_set.shape)
    def get_samples(self,num_iters,fixed_samples):return self._sample_iter[:num_iters] if fixed_samples else self._sample_set
    def generate_noise(self,shape,base_seed=None):return self.sample_lib.get_samples(shape,base_seed=base_seed)
    @property
    def full_scale_tril(self):return torch.diag_embed(self.scale_tril)
    @property
    def full_inv_cov(self):return torch.diag_embed(self.inv_cov)
    def shift(self,shift_steps,repeat_last):
        self.mean=torch.roll(self.mean,-shift_steps,-2)
        self.mean[:,-shift_steps:]=self.mean[:,-shift_steps-1:-shift_steps] if repeat_last else 0
    def reset_seed(self):self.sample_lib.reset_seed()
