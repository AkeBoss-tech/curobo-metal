from dataclasses import dataclass
from typing import Union,List,Callable
import torch
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
@dataclass
class LineSearchContext:
    device_cfg: DeviceCfg; line_search_scale: Union[List[float],torch.Tensor]; line_search_c_1: float; line_search_c_2: float
    num_problems: int; opt_dim: int; action_horizon: int; action_dim: int; step_scale: float; fix_terminal_action: bool
    action_horizon_step_max: torch.Tensor; use_cuda_kernel_line_search: bool; compute_costs_and_gradients: Callable
    convergence_iteration: int; cost_delta_threshold: float; cost_relative_threshold: float
    def __post_init__(self):
        if self.use_cuda_kernel_line_search: raise NotImplementedError("CUDA line-search kernels are unavailable on CPU/MPS")
        self.line_search_scale=torch.as_tensor(self.line_search_scale,device=self.device_cfg.device,dtype=self.device_cfg.dtype)
    @property
    def n_linesearch(self): return self.line_search_scale.numel()
    def update_num_problems(self,num_problems): self.num_problems=num_problems
    def _create_box_line_search(self,line_search_scale): self.line_search_scale=torch.as_tensor(line_search_scale,device=self.device_cfg.device,dtype=self.device_cfg.dtype)
