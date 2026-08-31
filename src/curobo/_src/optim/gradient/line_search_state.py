from dataclasses import dataclass
import torch
from curobo._src.util.logging import log_and_raise
@dataclass
class LineSearchState:
    action: torch.Tensor
    cost: torch.Tensor
    gradient: torch.Tensor
    idxs: torch.Tensor
