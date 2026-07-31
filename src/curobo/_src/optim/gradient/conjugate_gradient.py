from dataclasses import dataclass
import torch
from .gradient_descent import GradientDescentOptCfg,GradientDescentOpt
def jit_cg_compute_step_direction(grad,prev_grad,prev_step,max_beta,method):
    y=grad-prev_grad; key=method.lower()
    if "fletcher" in key: beta=grad.square().sum(-1)/(prev_grad.square().sum(-1).clamp_min(torch.finfo(grad.dtype).eps))
    else: beta=(grad*y).sum(-1)/(prev_grad.square().sum(-1).clamp_min(torch.finfo(grad.dtype).eps))
    beta=beta.clamp(0,max_beta)
    return -grad+beta.unsqueeze(-1)*prev_step
def jit_cg_shift_buffers(prev_grad,prev_step,shift_steps,action_dim):
    n=shift_steps*action_dim
    return torch.roll(prev_grad,-n,-1),torch.roll(prev_step,-n,-1)
@dataclass
class ConjugateGradientOptCfg(GradientDescentOptCfg):
    solver_type: str="conjugate_gradient"; solver_name: str="conjugate_gradient"; max_beta: float=1.0; beta_type: str="polak_ribiere"
class ConjugateGradientOpt(GradientDescentOpt): pass
