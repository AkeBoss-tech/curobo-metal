"""Portable per-problem quasi-Newton history."""
from typing import Optional

import torch

from curobo._src.optim.gradient.lbfgs_jit_helpers import (
    jit_lbfgs_update_buffers,
    lbfgs_shift_buffers_jit,
)
from curobo._src.types.device_cfg import DeviceCfg


class _QuasiNewtonBuffersPortableMixin:
    def __init__(self, device_cfg=None, n_problems=1, m=7, opt_dim=1, **kwargs):
        del kwargs
        self.device_cfg = device_cfg or DeviceCfg()
        self.m = m; self.resize(n_problems, opt_dim)
    def resize(self, num_problems, opt_dim):
        tensor_args = {
            "device": self.device_cfg.device,
            "dtype": self.device_cfg.dtype,
        }
        self.x_0 = torch.zeros(num_problems, opt_dim, **tensor_args)
        self.grad_0 = torch.zeros_like(self.x_0)
        self.s_buffer = torch.zeros(num_problems, self.m, opt_dim)
        self.s_buffer = self.s_buffer.to(**tensor_args)
        self.y_buffer = torch.zeros_like(self.s_buffer)
        self.rho_buffer = torch.zeros(num_problems, self.m, **tensor_args)
        return self
    @property
    def s(self):
        return self.s_buffer.permute(1, 0, 2)
    @property
    def y(self):
        return self.y_buffer.permute(1, 0, 2)
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


class QuasiNewtonBuffers(_QuasiNewtonBuffersPortableMixin):
    """Pinned L-BFGS buffer declaration using the portable tensor history."""

    def __init__(self, device_cfg: DeviceCfg, history: int):
        _QuasiNewtonBuffersPortableMixin.__init__(
            self, device_cfg=device_cfg, n_problems=1, m=history, opt_dim=1
        )
        self.device_cfg = device_cfg
        self.history = history

    def resize(self, num_problems: int, opt_dim: int):
        return _QuasiNewtonBuffersPortableMixin.resize(self, num_problems, opt_dim)

    def clear(self, mask: Optional[torch.Tensor] = None):
        return _QuasiNewtonBuffersPortableMixin.clear(self, mask)

    def set_reference(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        return _QuasiNewtonBuffersPortableMixin.set_reference(self, x, grad, mask)

    def update(self, q: torch.Tensor, grad_q: torch.Tensor):
        return _QuasiNewtonBuffersPortableMixin.update(self, q, grad_q)

    def shift(self, shift_steps: int, action_dim: int):
        return _QuasiNewtonBuffersPortableMixin.shift(self, shift_steps, action_dim)
