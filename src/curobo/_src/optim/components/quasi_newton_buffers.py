"""Portable per-problem quasi-Newton history."""
import torch

class QuasiNewtonBuffers:
    def __init__(self, device_cfg=None, n_problems=1, m=7, opt_dim=1, **kwargs):
        del device_cfg, kwargs
        self.m = m; self.resize(n_problems, opt_dim)
    def resize(self, num_problems, opt_dim):
        self.x_0 = torch.zeros(num_problems, opt_dim)
        self.grad_0 = torch.zeros_like(self.x_0)
        self.s_buffer = torch.zeros(num_problems, self.m, opt_dim)
        self.y_buffer = torch.zeros_like(self.s_buffer)
        self.rho_buffer = torch.zeros(num_problems, self.m)
        return self
    def clear(self, mask=None):
        tensors=(self.x_0,self.grad_0,self.s_buffer,self.y_buffer,self.rho_buffer)
        if mask is None:
            for x in tensors: x.zero_()
        else:
            for x in tensors: x[mask].zero_()
    reset = clear
    def set_reference(self, x, grad, mask=None):
        if mask is None: self.x_0.copy_(x); self.grad_0.copy_(grad)
        else: self.x_0[mask]=x[mask]; self.grad_0[mask]=grad[mask]
    def update(self, q, grad_q):
        s, y = q-self.x_0, grad_q-self.grad_0
        self.s_buffer=torch.roll(self.s_buffer,1,1); self.y_buffer=torch.roll(self.y_buffer,1,1)
        self.rho_buffer=torch.roll(self.rho_buffer,1,1)
        self.s_buffer[:,0]=s; self.y_buffer[:,0]=y
        self.rho_buffer[:,0]=1.0/(s.mul(y).sum(-1)).clamp_min(torch.finfo(q.dtype).eps)
        self.set_reference(q,grad_q)
    def shift(self, shift_steps, action_dim):
        n=shift_steps*action_dim
        for x in (self.x_0,self.grad_0,self.s_buffer,self.y_buffer):
            x.copy_(torch.roll(x,-n,-1)); x[..., -n:]=0
