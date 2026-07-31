from dataclasses import dataclass
import torch
@dataclass
class LineSearchState:
    action: torch.Tensor
    cost: torch.Tensor
    gradient: torch.Tensor
    idxs: torch.Tensor
