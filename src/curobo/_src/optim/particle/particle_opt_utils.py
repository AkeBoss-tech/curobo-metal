"""Portable tensor helpers shared by particle optimizers."""
from enum import Enum
import torch

class SquashType(Enum):
    CLAMP="CLAMP"; TANH="TANH"

def scale_ctrl(ctrl, action_lows, action_highs, squash_fn=SquashType.CLAMP):
    mode=getattr(squash_fn,"name",str(squash_fn)).upper()
    if "TANH" in mode:
        return action_lows + (torch.tanh(ctrl)+1)*0.5*(action_highs-action_lows)
    return torch.clamp(ctrl, min=action_lows, max=action_highs)

def gaussian_entropy(cov=None, L=None):
    if L is None:
        if cov is None: raise ValueError("cov or L is required")
        L=torch.linalg.cholesky(cov)
    n=L.shape[-1]
    return 0.5*n*(1.0+torch.log(torch.as_tensor(2*torch.pi,device=L.device,dtype=L.dtype))) + torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1)

def cost_to_go(cost_seq, gamma_seq, only_first=False):
    if gamma_seq is None: gamma_seq=torch.ones(cost_seq.shape[-1],device=cost_seq.device,dtype=cost_seq.dtype)
    weighted=cost_seq*gamma_seq
    result=torch.flip(torch.cumsum(torch.flip(weighted,[-1]),-1),[-1])
    return result[...,0] if only_first else result

def matrix_cholesky(A): return torch.linalg.cholesky(A)
def batch_cholesky(A): return torch.linalg.cholesky(A)
