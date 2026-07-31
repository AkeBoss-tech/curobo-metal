import torch
import torch.nn.functional as F
class KnotParticleProcessor:
    def __init__(self,horizon,action_dim,n_knots=3,device_cfg=None,degree=3,**kwargs):
        del device_cfg,kwargs; self.input_horizon=n_knots; self.horizon=horizon; self.action_dim=action_dim; self.input_ndims=n_knots*action_dim; self.degree=degree
    @staticmethod
    def bspline(c_arr,t_arr=None,n=100,degree=3):
        del t_arr,degree
        x=c_arr.movedim(-2,-1)
        return F.interpolate(x,size=n,mode="linear",align_corners=True).movedim(-1,-2)
    def process_samples(self,samples,filter_smooth=False):
        del filter_smooth
        return self.bspline(samples,n=self.horizon,degree=self.degree)
