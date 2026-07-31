"""Deterministic CPU-seeded samplers with device-resident results."""
import torch
from .processor_standard import StandardParticleProcessor
from .processor_knot import KnotParticleProcessor
from .processor_stomp import StompParticleProcessor

class ParticleSampler:
    def __init__(self,sample_config,horizon,action_dim,generator=None,post_processor=None):
        self.sample_config=sample_config; self.horizon=horizon; self.action_dim=action_dim; self.generator=generator
        self.post_processor=post_processor or StandardParticleProcessor(horizon,action_dim,sample_config.device_cfg,sample_config.filter_coeffs)
        self.samples=None; self.sample_shape=None
    def reset_seed(self): self.samples=None; self.sample_shape=None
    def get_samples(self,sample_shape,base_seed=None,filter_smooth=False,**kwargs):
        del kwargs
        count=int(sample_shape[0])
        if self.samples is None or self.sample_shape!=tuple(sample_shape) or not self.sample_config.fixed_samples:
            gen=torch.Generator(device="cpu"); gen.manual_seed(self.sample_config.seed if base_seed is None else base_seed)
            raw=torch.randn((count,self.post_processor.input_horizon,self.action_dim),generator=gen)
            cfg=self.sample_config.device_cfg
            raw=raw.to(device=getattr(cfg,"device","cpu"),dtype=getattr(cfg,"dtype",torch.float32))
            self.samples=self.post_processor.process_samples(raw,filter_smooth); self.sample_shape=tuple(sample_shape)
        return self.samples
    @classmethod
    def create_halton_particle_sampler(cls,cfg,horizon,action_dim): return cls(cfg,horizon,action_dim)
    @classmethod
    def create_random_particle_sampler(cls,cfg,horizon,action_dim): return cls(cfg,horizon,action_dim)
    @classmethod
    def create_knot_particle_sampler(cls,cfg,horizon,action_dim,sequencer_type="halton"):
        del sequencer_type
        return cls(cfg,horizon,action_dim,post_processor=KnotParticleProcessor(horizon,action_dim,cfg.n_knots,cfg.device_cfg,cfg.degree))
    @classmethod
    def create_stomp_particle_sampler(cls,cfg,horizon,action_dim):
        return cls(cfg,horizon,action_dim,post_processor=StompParticleProcessor(horizon,action_dim,cfg.device_cfg,cfg.stencil_type))

def create_particle_sampler(sample_type,sample_config,horizon,action_dim,**kwargs):
    del kwargs
    key=sample_type.lower()
    if "stomp" in key: return ParticleSampler.create_stomp_particle_sampler(sample_config,horizon,action_dim)
    if "knot" in key: return ParticleSampler.create_knot_particle_sampler(sample_config,horizon,action_dim)
    if key in ("halton","random"): return ParticleSampler(sample_config,horizon,action_dim)
    raise ValueError(f"unknown sample_type: {sample_type}")

class MixedParticleSampler:
    def __init__(self,sample_config,horizon,action_dim):
        self.sample_config=sample_config; self.samplers=[(ratio,create_particle_sampler(name,sample_config,horizon,action_dim)) for name,ratio in sample_config.sample_ratio.items() if ratio>0]
    def reset_seed(self):
        for _,s in self.samplers:s.reset_seed()
    def get_samples(self,sample_shape,base_seed=None,**kwargs):
        n=int(sample_shape[0]); counts=[int(n*r) for r,_ in self.samplers]
        if counts: counts[0]+=n-sum(counts)
        return torch.cat([s.get_samples([c],base_seed=base_seed,**kwargs) for c,(_,s) in zip(counts,self.samplers) if c],0)
