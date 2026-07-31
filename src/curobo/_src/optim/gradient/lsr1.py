import torch
from .lbfgs import LBFGSOpt
def jit_lsr1_compute_step_direction(y_buffer,s_buffer,grad,m,epsilon,stable_mode,hessian_0):
    del m,stable_mode
    direction=hessian_0*grad
    for y,s in zip(y_buffer.unbind(-2),s_buffer.unbind(-2)):
        u=s-hessian_0*y; denom=(u*y).sum(-1)
        safe=denom.abs()>epsilon
        direction=direction+torch.where(safe[:,None],u*(u*grad).sum(-1,keepdim=True)/denom[:,None],0)
    return -direction
class LSR1Opt(LBFGSOpt): pass
