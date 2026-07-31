import torch
class StandardParticleProcessor:
    def __init__(self,horizon,action_dim,device_cfg=None,filter_coeffs=None):
        self.input_horizon=self.horizon=horizon; self.action_dim=action_dim
        self.input_ndims=horizon*action_dim; self.filter_coeffs=filter_coeffs
    def _filter_samples(self,eps):
        if not self.filter_coeffs: return eps
        out=eps.clone(); coeff=torch.as_tensor(self.filter_coeffs,device=eps.device,dtype=eps.dtype)
        for t in range(1,eps.shape[-2]):
            out[...,t,:]=coeff[-1]*eps[...,t,:]+coeff[:-1].sum()*out[...,t-1,:]
        return out
    def _filter_smooth(self,samples): return self._filter_samples(samples)
    def process_samples(self,samples,filter_smooth=False): return self._filter_samples(samples) if filter_smooth or self.filter_coeffs else samples
