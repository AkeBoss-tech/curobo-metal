import torch
from .stomp_covariance import get_stomp_cov
class StompParticleProcessor:
    def __init__(self,horizon,action_dim,device_cfg=None,stencil_type="3point",**kwargs):
        del kwargs; self.input_horizon=self.horizon=horizon; self.action_dim=action_dim; self.input_ndims=horizon*action_dim
        cov=get_stomp_cov(horizon,stencil_type=stencil_type)
        device=getattr(device_cfg,"device","cpu"); dtype=getattr(device_cfg,"dtype",torch.float32)
        self.factor=torch.linalg.cholesky(cov.to(device=device,dtype=dtype)+torch.eye(horizon,device=device,dtype=dtype)*1e-6)
    def process_samples(self,samples,filter_smooth=False):
        del filter_smooth
        return torch.einsum("ij,...jd->...id",self.factor,samples)
