"""Pinned L-BFGS surface backed by portable batched L-BFGS."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class LBFGSOptCfg(PortableOptCfg):
    solver_type: str = "lbfgs"
    solver_name: str = "lbfgs"
    inner_iters: int = 25
    _num_rollout_instances: int = 2
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    fixed_iters: bool = True
    convergence_iteration: int = 0
    minimum_iters: Optional[int] = None
    return_best_action: bool = True
    line_search_scale: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])
    line_search_type: Any = "APPROX_WOLFE"
    use_cuda_kernel_line_search: bool = False
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9
    history: int = 7
    epsilon: float = 0.01
    use_cuda_kernel_step_direction: bool = False
    stable_mode: bool = True
    use_cuda_kernel_shared_buffers: bool = False
    initial_step_scale: float = 0.1

    @property
    def portable_line_search(self):
        name = getattr(self.line_search_type, "name", str(self.line_search_type)).lower()
        return "strong_wolfe" if "strong" in name else "armijo"


class LBFGSOpt(PortableOptimizer):
    strategy = "lbfgs"


__all__ = ["LBFGSOptCfg", "LBFGSOpt"]
