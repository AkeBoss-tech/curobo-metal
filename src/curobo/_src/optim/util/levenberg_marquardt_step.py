from dataclasses import dataclass
from typing import Optional
import torch
@dataclass
class LevenbergMarquardtState:
    jacobian: torch.Tensor; jTerror: torch.Tensor; lambda_damping: torch.Tensor
    joint_position_in: torch.Tensor; joint_position_out: torch.Tensor; pred_reduction: torch.Tensor
    _batch_size: Optional[int]=None; _action_dim: Optional[int]=None; _n_residuals: Optional[int]=None
    @property
    def batch_size(self): return self._batch_size or self.jacobian.shape[0]
    @property
    def action_dim(self): return self._action_dim or self.jacobian.shape[-1]
    @property
    def n_residuals(self): return self._n_residuals or self.jacobian.shape[-2]
class LevenbergMarquardtStep:
    def __init__(self,action_dim,n_residuals,tile_threads=128,**kwargs):
        del kwargs;self._action_dim=action_dim;self._n_residuals=n_residuals;self._tile_threads=tile_threads
    @property
    def action_dim(self):return self._action_dim
    @property
    def n_residuals(self):return self._n_residuals
    @property
    def tile_threads(self):return self._tile_threads
    @staticmethod
    def create_lm_warp_kernel(dof,n_res): raise NotImplementedError("Warp LM kernels are unavailable; call the portable torch step")
    def __call__(self,state):
        j=state.jacobian; eye=torch.eye(self.action_dim,device=j.device,dtype=j.dtype)
        lhs=j.transpose(-1,-2)@j+state.lambda_damping.reshape(-1,1,1)*eye
        delta=torch.linalg.solve(lhs,state.jTerror.unsqueeze(-1)).squeeze(-1)
        state.joint_position_out.copy_(state.joint_position_in-delta)
        state.pred_reduction.copy_((delta*state.jTerror).sum(-1))
        return state.joint_position_out
